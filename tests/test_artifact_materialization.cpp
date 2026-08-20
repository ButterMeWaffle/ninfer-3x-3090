#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};

void require(bool condition, const char* message);

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
        },
        "materialization");
}

ninfer::test::artifact_fixture::TemporaryArtifact write_shard_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    using namespace ninfer::artifact;
    constexpr std::array<std::uint64_t, 2> gate_shape = {8, 128};
    constexpr std::array<std::uint64_t, 2> down_shape = {4, 256};
    const RowSplitGeometry gate = row_split_geometry(NumericFormat::Q4G64_F16S, gate_shape);
    const RowSplitGeometry down = row_split_geometry(NumericFormat::Q5G64_F16S, down_shape);
    const std::uint64_t down_offset = ninfer::test::artifact_fixture::align_up(
        gate.encoded_bytes, tensor_alignment(StorageLayout::RowSplitK128V1));
    auto fixture = ninfer::test::artifact_fixture::write_fixture(
        {{"identity", {{"model_id", "shard-model"}, {"weights_id", "shard-weights"}}},
         {"objects",
          Json::array({
              {{"name", "weights/gate_up"},
               {"kind", "tensor"},
               {"shape", {8, 128}},
               {"format", "Q4G64_F16S"},
               {"layout", "row-split-k128-v1"},
               {"offset", 0},
               {"bytes", gate.encoded_bytes}},
              {{"name", "weights/down"},
               {"kind", "tensor"},
               {"shape", {4, 256}},
               {"format", "Q5G64_F16S"},
               {"layout", "row-split-k128-v1"},
               {"offset", down_offset},
               {"bytes", down.encoded_bytes}},
          })}},
        "shard_materialization");

    Reader reader(fixture.path);
    std::fstream output(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
    require(static_cast<bool>(output), "failed to reopen shard fixture");
    for (std::size_t object = 0; object < reader.objects().size(); ++object) {
        const PayloadSpan payload = reader.payload(reader.objects()[object]);
        std::vector<std::byte> pattern(payload.data.size());
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = std::byte{static_cast<unsigned char>((i * 37 + object * 53 + 11) % 251)};
        }
        output.seekp(static_cast<std::streamoff>(payload.absolute_offset));
        output.write(reinterpret_cast<const char*>(pattern.data()),
                     static_cast<std::streamsize>(pattern.size()));
    }
    require(static_cast<bool>(output), "failed to populate shard fixture");
    return fixture;
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<int> find_peer_pair(int count) {
    for (int first = 0; first < count; ++first) {
        cudaDeviceProp first_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&first_properties, first));
        for (int second = first + 1; second < count; ++second) {
            cudaDeviceProp second_properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&second_properties, second));
            int first_to_second = 0;
            int second_to_first = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&first_to_second, first, second));
            CUDA_CHECK(cudaDeviceCanAccessPeer(&second_to_first, second, first));
            if (first_to_second != 0 && second_to_first != 0 &&
                first_properties.major == second_properties.major &&
                first_properties.minor == second_properties.minor) {
                return {first, second};
            }
        }
    }
    return {};
}

std::vector<std::byte> copy_device_bytes(const void* source, std::size_t bytes) {
    std::vector<std::byte> out(bytes);
    CUDA_CHECK(cudaMemcpy(out.data(), source, bytes, cudaMemcpyDeviceToHost));
    return out;
}

void require_row_split_region(const std::vector<std::byte>& destination,
                              const ninfer::artifact::RowSplitGeometry& destination_geometry,
                              std::uint64_t destination_row,
                              const std::vector<std::byte>& source,
                              const ninfer::artifact::RowSplitGeometry& source_geometry,
                              std::uint64_t source_row, std::uint64_t source_group,
                              std::uint64_t rows, std::uint64_t groups, const char* message) {
    struct Plane {
        std::uint64_t source_offset;
        std::uint64_t destination_offset;
        std::uint64_t bytes_per_group;
    };
    const std::array planes = {
        Plane{0, 0, source_geometry.low_bytes_per_group},
        Plane{source_geometry.high_plane_offset, destination_geometry.high_plane_offset,
              source_geometry.high_bytes_per_group},
        Plane{source_geometry.scale_plane_offset, destination_geometry.scale_plane_offset, 2},
    };
    for (const Plane& plane : planes) {
        if (plane.bytes_per_group == 0) { continue; }
        const std::uint64_t source_pitch =
            source_geometry.groups_per_row * plane.bytes_per_group;
        const std::uint64_t destination_pitch =
            destination_geometry.groups_per_row * plane.bytes_per_group;
        const std::uint64_t width = groups * plane.bytes_per_group;
        for (std::uint64_t row = 0; row < rows; ++row) {
            const auto* expected =
                source.data() + plane.source_offset + (source_row + row) * source_pitch +
                source_group * plane.bytes_per_group;
            const auto* actual = destination.data() + plane.destination_offset +
                                 (destination_row + row) * destination_pitch;
            require(std::equal(expected, expected + width, actual), message);
        }
    }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 3 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 2 && plan.device_capacity_bytes == 260,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 2 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kSecondTensor.size(),
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        const std::vector<int> peer_pair = find_peer_pair(device_count);
        if (!peer_pair.empty()) {
            ninfer::DeviceContext dual{std::span<const int>(peer_pair)};
            auto replicated = ninfer::artifact::materialize(reader, plan, dual);
            require(replicated.device_arena_count() == 2,
                    "peer materialization did not create two device-local arenas");
            const auto tensor_offset =
                static_cast<const std::byte*>(replicated.device_data(tensor)) -
                static_cast<const std::byte*>(replicated.device_arena().base());
            const auto* replica_tensor =
                static_cast<const std::byte*>(replicated.device_arena(1).base()) + tensor_offset;
            std::array<std::byte, kTensor.size()> replica_copied{};
            {
                ninfer::ScopedDeviceRank secondary(dual, 1);
                CUDA_CHECK(cudaMemcpy(replica_copied.data(), replica_tensor, replica_copied.size(),
                                      cudaMemcpyDeviceToHost));
            }
            require(replica_copied == kTensor,
                    "secondary device-local tensor replica differs from the artifact");

            auto shard_fixture = write_shard_fixture();
            ninfer::artifact::Reader shard_reader(shard_fixture.path);
            ninfer::artifact::Binder shard_binder(shard_reader);
            constexpr std::array<std::uint64_t, 2> gate_shape = {8, 128};
            constexpr std::array<std::uint64_t, 2> down_shape = {4, 256};
            const auto gate = shard_binder.require_tensor(
                "weights/gate_up", ninfer::artifact::NumericFormat::Q4G64_F16S,
                ninfer::artifact::StorageLayout::RowSplitK128V1, gate_shape);
            shard_binder.materialize_on_device(gate);
            shard_binder.shard_row_split_across_devices(
                gate, ninfer::artifact::RowSplitShardAxis::PairedRows, 2);
            const auto down = shard_binder.require_tensor(
                "weights/down", ninfer::artifact::NumericFormat::Q5G64_F16S,
                ninfer::artifact::StorageLayout::RowSplitK128V1, down_shape);
            shard_binder.materialize_on_device(down);
            shard_binder.shard_row_split_across_devices(
                down, ninfer::artifact::RowSplitShardAxis::Columns, 128);
            const auto shard_plan = shard_binder.finish();
            require(shard_plan.row_split_shards.size() == 2 &&
                        shard_plan.source_device_capacity_bytes == 1568 &&
                        shard_plan.device_capacity_bytes == 1040 &&
                        shard_plan.secondary_device_capacity_bytes == 1040,
                    "binder produced the wrong cross-device shard plan");

            const auto gate_payload = shard_reader.payload(shard_reader.objects().at(gate.index));
            const auto down_payload = shard_reader.payload(shard_reader.objects().at(down.index));
            const std::vector<std::byte> gate_source(gate_payload.data.begin(),
                                                     gate_payload.data.end());
            const std::vector<std::byte> down_source(down_payload.data.begin(),
                                                     down_payload.data.end());
            auto sharded = ninfer::artifact::materialize(shard_reader, shard_plan, dual);
            require(sharded.device_arena_count() == 2 &&
                        sharded.device_arena().capacity() == 1040 &&
                        sharded.device_arena(1).capacity() == 1040 &&
                        sharded.has_device_data(gate, 1) && sharded.has_device_data(down, 1),
                    "cross-device compact arenas have the wrong geometry");

            std::vector<std::byte> primary_gate;
            std::vector<std::byte> primary_down;
            {
                ninfer::ScopedDeviceRank primary(dual, 0);
                primary_gate = copy_device_bytes(sharded.device_data(gate, 0), 272);
                primary_down = copy_device_bytes(sharded.device_data(down, 0), 528);
            }
            std::vector<std::byte> secondary_gate;
            std::vector<std::byte> secondary_down;
            {
                ninfer::ScopedDeviceRank secondary(dual, 1);
                secondary_gate = copy_device_bytes(sharded.device_data(gate, 1), 272);
                secondary_down = copy_device_bytes(sharded.device_data(down, 1), 528);
            }

            const auto source_gate_geometry = ninfer::artifact::row_split_geometry(
                ninfer::artifact::NumericFormat::Q4G64_F16S, gate_shape);
            constexpr std::array<std::uint64_t, 2> compact_gate_shape = {4, 128};
            const auto compact_gate_geometry = ninfer::artifact::row_split_geometry(
                ninfer::artifact::NumericFormat::Q4G64_F16S, compact_gate_shape);
            require_row_split_region(primary_gate, compact_gate_geometry, 0, gate_source,
                                     source_gate_geometry, 0, 0, 2, 2,
                                     "primary compact gate rows differ from the artifact");
            require_row_split_region(primary_gate, compact_gate_geometry, 2, gate_source,
                                     source_gate_geometry, 4, 0, 2, 2,
                                     "primary compact up rows differ from the artifact");
            require_row_split_region(secondary_gate, compact_gate_geometry, 0, gate_source,
                                     source_gate_geometry, 2, 0, 2, 2,
                                     "secondary compact gate rows differ from the artifact");
            require_row_split_region(secondary_gate, compact_gate_geometry, 2, gate_source,
                                     source_gate_geometry, 6, 0, 2, 2,
                                     "secondary compact up rows differ from the artifact");

            const auto source_down_geometry = ninfer::artifact::row_split_geometry(
                ninfer::artifact::NumericFormat::Q5G64_F16S, down_shape);
            constexpr std::array<std::uint64_t, 2> compact_down_shape = {4, 128};
            const auto compact_down_geometry = ninfer::artifact::row_split_geometry(
                ninfer::artifact::NumericFormat::Q5G64_F16S, compact_down_shape);
            require_row_split_region(primary_down, compact_down_geometry, 0, down_source,
                                     source_down_geometry, 0, 0, 4, 2,
                                     "primary compact down columns differ from the artifact");
            require_row_split_region(secondary_down, compact_down_geometry, 0, down_source,
                                     source_down_geometry, 0, 2, 4, 2,
                                     "secondary compact down columns differ from the artifact");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
