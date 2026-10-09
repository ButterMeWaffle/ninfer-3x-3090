#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

const RowSplitShardMaterialization*
find_row_split_shard(const MaterializationPlan& plan, ObjectHandle object) {
    const auto it = std::find_if(
        plan.row_split_shards.begin(), plan.row_split_shards.end(),
        [&](const RowSplitShardMaterialization& shard) {
            return shard.object.index == object.index;
        });
    return it == plan.row_split_shards.end() ? nullptr : &*it;
}

DeviceSpan allocate_planned(DeviceArena& arena, std::uint64_t offset, std::uint64_t bytes,
                            std::uint64_t alignment, const char* label) {
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(SIZE_MAX) ||
        alignment == 0 || alignment > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError(std::string(label) + " has invalid allocation geometry");
    }
    const DeviceSpan storage = arena.alloc_bytes(static_cast<std::size_t>(bytes),
                                                  static_cast<std::size_t>(alignment));
    const auto actual_offset =
        static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                   static_cast<std::byte*>(arena.base()));
    if (actual_offset != offset) {
        throw ArtifactError(std::string(label) + " offset does not match its packed plan");
    }
    return storage;
}

void copy_peer_2d(void* destination, int destination_device, std::size_t destination_pitch,
                  const void* source, int source_device, std::size_t source_pitch,
                  std::size_t width, std::size_t height, cudaStream_t stream) {
    if (width == 0 || height == 0 || width > source_pitch || width > destination_pitch) {
        throw ArtifactError("row-split peer copy has invalid pitch geometry");
    }
    cudaMemcpy3DPeerParms params{};
    params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(source), source_pitch, width, height);
    params.srcDevice = source_device;
    params.dstPtr = make_cudaPitchedPtr(destination, destination_pitch, width, height);
    params.dstDevice = destination_device;
    params.extent = make_cudaExtent(width, height, 1);
    CUDA_CHECK(cudaMemcpy3DPeerAsync(&params, stream));
}

void copy_row_split_region(std::byte* destination, const RowSplitGeometry& destination_geometry,
                           std::uint64_t destination_row, const std::byte* source,
                           const RowSplitGeometry& source_geometry, std::uint64_t source_row,
                           std::uint64_t source_group, std::uint64_t rows,
                           std::uint64_t groups, int destination_device, int source_device,
                           cudaStream_t stream, std::uint64_t& copied_bytes) {
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
        const std::uint64_t source_offset =
            plane.source_offset + source_row * source_pitch +
            source_group * plane.bytes_per_group;
        const std::uint64_t destination_offset =
            plane.destination_offset + destination_row * destination_pitch;
        copy_peer_2d(destination + destination_offset, destination_device,
                     static_cast<std::size_t>(destination_pitch), source + source_offset,
                     source_device, static_cast<std::size_t>(source_pitch),
                     static_cast<std::size_t>(width), static_cast<std::size_t>(rows), stream);
        copied_bytes = checked_add(copied_bytes, width * rows,
                                   "peer copied byte count overflows u64");
    }
}

void copy_row_split_shard(const Reader& reader, const RowSplitShardMaterialization& shard,
                          const void* source_storage, int source_device,
                          const std::vector<void*>& rank_storage,
                          const std::vector<int>& rank_devices,
                          cudaStream_t stream, std::uint64_t& copied_bytes) {
    const auto* tensor =
        std::get_if<TensorDescriptor>(&reader.objects().at(shard.object.index));
    if (tensor == nullptr || tensor->layout != StorageLayout::RowSplitK128V1 ||
        tensor->shape.size() != 2) {
        throw ArtifactError("cross-device shard does not describe a row-split tensor");
    }
    const RowSplitGeometry source_geometry = row_split_geometry(tensor->format, tensor->shape);
    const std::uint64_t rows = tensor->shape[0];
    const std::uint64_t cols = tensor->shape[1];
    const std::size_t rank_count = shard.splits.size() + 1;

    const auto rank_end = [&](std::size_t rank) -> std::uint64_t {
        if (rank < shard.splits.size()) { return shard.splits[rank]; }
        return (shard.axis == RowSplitShardAxis::PairedRows) ? rows / 2 : cols;
    };

    const auto* source = static_cast<const std::byte*>(source_storage);

    if (shard.axis == RowSplitShardAxis::PairedRows) {
        const std::uint64_t intermediate = rows / 2;
        for (std::size_t rank = 0; rank < rank_count; ++rank) {
            const std::uint64_t begin  = rank == 0 ? 0 : shard.splits[rank - 1];
            const std::uint64_t end    = rank_end(rank);
            const std::uint64_t extent = end - begin;
            const RowSplitGeometry rank_geometry =
                row_split_geometry(tensor->format, std::array<std::uint64_t, 2>{2 * extent, cols});
            auto* dest = static_cast<std::byte*>(rank_storage[rank]);
            const int dev = rank_devices[rank];
            copy_row_split_region(dest, rank_geometry, 0, source, source_geometry,
                                  begin, 0, extent, source_geometry.groups_per_row,
                                  dev, source_device, stream, copied_bytes);
            copy_row_split_region(dest, rank_geometry, extent, source, source_geometry,
                                  intermediate + begin, 0, extent, source_geometry.groups_per_row,
                                  dev, source_device, stream, copied_bytes);
        }
        return;
    }

    std::uint64_t group_offset = 0;
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        const std::uint64_t begin  = rank == 0 ? 0 : shard.splits[rank - 1];
        const std::uint64_t end    = rank_end(rank);
        const std::uint64_t extent = end - begin;
        const RowSplitGeometry rank_geometry =
            row_split_geometry(tensor->format, std::array<std::uint64_t, 2>{rows, extent});
        auto* dest = static_cast<std::byte*>(rank_storage[rank]);
        const int dev = rank_devices[rank];
        copy_row_split_region(dest, rank_geometry, 0, source, source_geometry,
                              0, group_offset, rows, rank_geometry.groups_per_row,
                              dev, source_device, stream, copied_bytes);
        group_offset += rank_geometry.groups_per_row;
    }
}

} // namespace

void MaterializedArtifact::release_device_arenas() noexcept {
    int saved_device = -1;
    (void)cudaGetDevice(&saved_device);
    for (std::size_t rank = device_arenas_.size(); rank-- > 0;) {
        if (rank < device_arena_devices_.size()) {
            (void)cudaSetDevice(device_arena_devices_[rank]);
        }
        device_arenas_[rank].reset();
    }
    if (saved_device >= 0) { (void)cudaSetDevice(saved_device); }
    device_arenas_.clear();
    device_arena_devices_.clear();
}

MaterializedArtifact::~MaterializedArtifact() { release_device_arenas(); }

MaterializedArtifact::MaterializedArtifact(MaterializedArtifact&& other) noexcept
    : device_arenas_(std::move(other.device_arenas_)),
      device_arena_devices_(std::move(other.device_arena_devices_)),
      objects_(std::move(other.objects_)), stats_(other.stats_) {}

MaterializedArtifact& MaterializedArtifact::operator=(MaterializedArtifact&& other) noexcept {
    if (this == &other) { return *this; }
    release_device_arenas();
    device_arenas_       = std::move(other.device_arenas_);
    device_arena_devices_ = std::move(other.device_arena_devices_);
    objects_             = std::move(other.objects_);
    stats_               = other.stats_;
    return *this;
}

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    return device_data(handle, 0);
}

void* MaterializedArtifact::device_data(ObjectHandle handle, std::size_t rank) const {
    if (handle.index >= objects_.size() || rank >= objects_[handle.index].device.size() ||
        objects_[handle.index].device[rank] == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device[rank];
}

bool MaterializedArtifact::has_device_data(ObjectHandle handle, std::size_t rank) const noexcept {
    return handle.index < objects_.size() && rank < objects_[handle.index].device.size() &&
           objects_[handle.index].device[rank] != nullptr;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    return device_arena(0);
}

DeviceArena& MaterializedArtifact::device_arena(std::size_t rank) {
    if (rank >= device_arenas_.size() || !device_arenas_[rank]) {
        throw ArtifactError("artifact has no device tensor backing for requested rank");
    }
    return *device_arenas_[rank];
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t final_primary_capacity = plan.device_capacity_bytes;
    const std::uint64_t source_capacity = plan.source_device_capacity_bytes == 0
                                              ? final_primary_capacity
                                              : plan.source_device_capacity_bytes;
    if (source_capacity == 0 || source_capacity > static_cast<std::uint64_t>(SIZE_MAX) ||
        final_primary_capacity == 0 ||
        final_primary_capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    if (!plan.row_split_shards.empty() && !device.model_parallel()) {
        throw ArtifactError("cross-device weight shards require two CUDA devices");
    }
    out.device_arenas_.push_back(
        std::make_unique<DeviceArena>(static_cast<std::size_t>(source_capacity)));
    out.device_arena_devices_.push_back(device.device_ids()[0]);
    if (device.model_parallel() && plan.row_split_shards.empty()) {
        for (std::size_t rank = 1; rank < device.size(); ++rank) {
            ScopedDeviceRank r(device, rank);
            out.device_arenas_.push_back(
                std::make_unique<DeviceArena>(static_cast<std::size_t>(source_capacity)));
            out.device_arena_devices_.push_back(device.device_ids()[rank]);
        }
    }
    out.stats_.device_capacity_bytes = final_primary_capacity;
    out.stats_.secondary_device_capacity_bytes =
        plan.row_split_shards.empty() && device.model_parallel()
            ? source_capacity
            : plan.secondary_device_capacity_bytes;
    out.stats_.tertiary_device_capacity_bytes =
        plan.row_split_shards.empty() && device.size() > 2
            ? source_capacity
            : plan.tertiary_device_capacity_bytes;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        DeviceSpan storage = out.device_arena().alloc_bytes(
            static_cast<std::size_t>(placement.bytes),
            static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena().base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device.resize(out.device_arena_count(), nullptr);
        out.objects_.at(placement.object.index).device[0] = storage.data;
        for (std::size_t rank = 1; rank < out.device_arena_count(); ++rank) {
            DeviceArena& replica = out.device_arena(rank);
            const DeviceSpan replica_storage = replica.alloc_bytes(
                static_cast<std::size_t>(placement.bytes),
                static_cast<std::size_t>(placement.alignment));
            const auto replica_offset =
                static_cast<std::uint64_t>(static_cast<std::byte*>(replica_storage.data) -
                                           static_cast<std::byte*>(replica.base()));
            if (replica_offset != placement.offset) {
                throw ArtifactError("replica materialization layout does not match primary");
            }
            out.objects_.at(placement.object.index).device[rank] = replica_storage.data;
        }
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
        });
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, device.load_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
            slot.pending = true;

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (device.model_parallel() && plan.row_split_shards.empty()) {
        // Replicate the immutable packed tensor arena once over the peer link to each secondary.
        const int primary_device = device.device_ids()[0];
        for (std::size_t rank = 1; rank < device.size(); ++rank) {
            const int peer_device = device.device_ids()[rank];
            {
                ScopedDeviceRank r(device, rank);
                CUDA_CHECK(cudaMemcpyPeerAsync(
                    out.device_arena(rank).base(), peer_device, out.device_arena().base(),
                    primary_device, static_cast<std::size_t>(source_capacity), device.stream));
                device.synchronize_rank(rank);
            }
            out.stats_.peer_to_peer_bytes = checked_add(
                out.stats_.peer_to_peer_bytes, source_capacity,
                "peer copied byte count overflows u64");
        }
    } else if (device.model_parallel()) {
        if (plan.secondary_device_capacity_bytes == 0 ||
            plan.secondary_device_capacity_bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("secondary shard backing size is invalid");
        }
        if (device.size() > 2 &&
            (plan.tertiary_device_capacity_bytes == 0 ||
             plan.tertiary_device_capacity_bytes > static_cast<std::uint64_t>(SIZE_MAX))) {
            throw ArtifactError("tertiary shard backing size is invalid");
        }

        const int primary_device   = device.device_ids()[0];
        const int secondary_device = device.device_ids()[1];
        const int tertiary_device  = device.size() > 2 ? device.device_ids()[2] : -1;
        std::vector<void*> staged_primary(plan.object_count, nullptr);
        MaterializedArtifact primary_staging;
        {
            ScopedDeviceRank secondary(device, 1);
            primary_staging.device_arenas_.push_back(
                std::make_unique<DeviceArena>(static_cast<std::size_t>(final_primary_capacity)));
            primary_staging.device_arena_devices_.push_back(secondary_device);
            out.device_arenas_.push_back(std::make_unique<DeviceArena>(
                static_cast<std::size_t>(plan.secondary_device_capacity_bytes)));
            out.device_arena_devices_.push_back(secondary_device);
            if (device.size() > 2) {
                ScopedDeviceRank tertiary(device, 2);
                out.device_arenas_.push_back(std::make_unique<DeviceArena>(
                    static_cast<std::size_t>(plan.tertiary_device_capacity_bytes)));
                out.device_arena_devices_.push_back(tertiary_device);
            }
            const std::size_t final_arena_count = out.device_arenas_.size();
            for (auto& obj : out.objects_) {
                obj.device.resize(final_arena_count, nullptr);
            }

            for (const DeviceMaterialization& placement : plan.device_objects) {
                const DeviceSpan primary_storage = allocate_planned(
                    primary_staging.device_arena(), placement.primary_offset,
                    placement.primary_bytes, placement.alignment, "primary shard");
                staged_primary.at(placement.object.index) = primary_storage.data;
                const RowSplitShardMaterialization* shard =
                    find_row_split_shard(plan, placement.object);
                if (shard == nullptr) {
                    // Non-sharded: replicate to each secondary.
                    CUDA_CHECK(cudaMemcpyPeerAsync(
                        primary_storage.data, secondary_device,
                        out.objects_.at(placement.object.index).device[0], primary_device,
                        static_cast<std::size_t>(placement.bytes), device.stream));
                    out.stats_.peer_to_peer_bytes = checked_add(
                        out.stats_.peer_to_peer_bytes, placement.bytes,
                        "peer copied byte count overflows u64");
                    if (device.size() > 2) {
                        CUDA_CHECK(cudaMemcpyPeerAsync(
                            out.objects_.at(placement.object.index).device[2], tertiary_device,
                            out.objects_.at(placement.object.index).device[0], primary_device,
                            static_cast<std::size_t>(placement.bytes), device.stream));
                        out.stats_.peer_to_peer_bytes = checked_add(
                            out.stats_.peer_to_peer_bytes, placement.bytes,
                            "peer copied byte count overflows u64");
                    }
                    continue;
                }

                // Sharded: allocate per-rank storage and copy from the full source.
                std::vector<void*> rank_storage(out.device_arena_count());
                std::vector<int> rank_devices(out.device_arena_count());
                rank_storage[0] = primary_storage.data;
                rank_devices[0] = secondary_device; // staged primary

                const DeviceSpan secondary_storage = allocate_planned(
                    out.device_arena(1), shard->secondary_offsets[0], shard->secondary_bytes[0],
                    placement.alignment, "secondary shard");
                rank_storage[1] = secondary_storage.data;
                rank_devices[1] = secondary_device;
                out.objects_.at(placement.object.index).device[1] = secondary_storage.data;

                if (device.size() > 2) {
                    const DeviceSpan tertiary_storage = allocate_planned(
                        out.device_arena(2), shard->secondary_offsets[1], shard->secondary_bytes[1],
                        placement.alignment, "tertiary shard");
                    rank_storage[2] = tertiary_storage.data;
                    rank_devices[2] = tertiary_device;
                    out.objects_.at(placement.object.index).device[2] = tertiary_storage.data;
                }

                copy_row_split_shard(
                    reader, *shard, out.objects_.at(placement.object.index).device[0],
                    primary_device, rank_storage, rank_devices, device.stream,
                    out.stats_.peer_to_peer_bytes);
            }
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }

        {
            ScopedDeviceRank primary(device, 0);
            out.device_arenas_[0].reset();
            out.device_arenas_[0] =
                std::make_unique<DeviceArena>(static_cast<std::size_t>(final_primary_capacity));
            for (const DeviceMaterialization& placement : plan.device_objects) {
                const DeviceSpan primary_storage = allocate_planned(
                    out.device_arena(), placement.primary_offset, placement.primary_bytes,
                    placement.alignment, "final primary shard");
                out.objects_.at(placement.object.index).device[0] = primary_storage.data;
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(
                out.device_arena().base(), primary_device,
                primary_staging.device_arena().base(), secondary_device,
                static_cast<std::size_t>(final_primary_capacity), device.stream));
            device.synchronize_rank(0);
            out.stats_.peer_to_peer_bytes = checked_add(
                out.stats_.peer_to_peer_bytes, final_primary_capacity,
                "peer copied byte count overflows u64");
        }
    }
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace ninfer::artifact
