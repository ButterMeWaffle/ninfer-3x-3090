#include "artifact/reader.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::RowSplitGeometry;

struct PackedRows {
    std::vector<std::uint8_t> bytes;
    RowSplitGeometry geometry;
    QType qtype;
    std::int32_t rows;
    std::int32_t columns;

    Weight device_weight(void* storage) const {
        auto* base = static_cast<std::uint8_t*>(storage);
        Weight out{};
        out.payload          = base;
        out.payload_bytes    = geometry.encoded_bytes;
        out.high_plane_bytes = geometry.high_plane_bytes;
        out.qtype            = qtype;
        out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
        out.qdata            = base;
        out.qhigh = geometry.high_plane_bytes == 0 ? nullptr : base + geometry.high_plane_offset;
        out.scales          = base + geometry.scale_plane_offset;
        out.n               = rows;
        out.k               = columns;
        out.group           = static_cast<std::int32_t>(geometry.group_size);
        out.layout          = QuantLayout::RowSplit;
        out.scale_dtype     = DType::FP16;
        out.ndim            = 2;
        out.shape[0]        = rows;
        out.shape[1]        = columns;
        out.padded_shape[0] = rows;
        out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
        return out;
    }
};

void fill_fp16_one(std::vector<std::uint8_t>& bytes, std::size_t begin, std::size_t count) {
    for (std::size_t offset = begin; offset < begin + count; offset += 2) {
        bytes[offset]     = 0x00;
        bytes[offset + 1] = 0x3c;
    }
}

PackedRows make_q4_gate_up(std::int32_t intermediate, std::int32_t logical_row_begin) {
    constexpr std::int32_t kInput = 5120;
    const std::array<std::uint64_t, 2> shape = {
        static_cast<std::uint64_t>(2 * intermediate), kInput};
    const RowSplitGeometry geometry =
        ninfer::artifact::row_split_geometry(NumericFormat::Q4G64_F16S, shape);
    PackedRows out{{}, geometry, QType::Q4G64_F16S, 2 * intermediate, kInput};
    out.bytes.resize(static_cast<std::size_t>(geometry.encoded_bytes));
    const std::size_t row_bytes = static_cast<std::size_t>(
        geometry.groups_per_row * geometry.low_bytes_per_group);
    for (std::int32_t row = 0; row < 2 * intermediate; ++row) {
        const bool up = row >= intermediate;
        const std::int32_t logical = logical_row_begin + row % intermediate;
        const std::uint8_t value =
            static_cast<std::uint8_t>(1 + logical % 3 + (up ? 2 : 0));
        std::fill_n(out.bytes.data() + static_cast<std::size_t>(row) * row_bytes, row_bytes,
                    static_cast<std::uint8_t>(value | (value << 4)));
    }
    fill_fp16_one(out.bytes, static_cast<std::size_t>(geometry.scale_plane_offset),
                  static_cast<std::size_t>(geometry.scale_plane_bytes));
    return out;
}

PackedRows make_q5_down(std::int32_t columns, std::int32_t logical_group_begin) {
    constexpr std::int32_t kRows = 5120;
    const std::array<std::uint64_t, 2> shape = {
        kRows, static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry =
        ninfer::artifact::row_split_geometry(NumericFormat::Q5G64_F16S, shape);
    PackedRows out{{}, geometry, QType::Q5G64_F16S, kRows, columns};
    out.bytes.resize(static_cast<std::size_t>(geometry.encoded_bytes));
    for (std::int32_t row = 0; row < kRows; ++row) {
        for (std::uint64_t group = 0; group < geometry.groups_per_row; ++group) {
            const std::uint8_t value = static_cast<std::uint8_t>(
                1 + (logical_group_begin + static_cast<std::int32_t>(group) + row) % 5);
            const std::size_t index =
                static_cast<std::size_t>(row) * geometry.groups_per_row + group;
            std::fill_n(out.bytes.data() + index * geometry.low_bytes_per_group,
                        static_cast<std::size_t>(geometry.low_bytes_per_group),
                        static_cast<std::uint8_t>(value | (value << 4)));
        }
    }
    fill_fp16_one(out.bytes, static_cast<std::size_t>(geometry.scale_plane_offset),
                  static_cast<std::size_t>(geometry.scale_plane_bytes));
    return out;
}

std::vector<std::uint16_t> constant_activation(std::int32_t rows, std::int32_t tokens) {
    return std::vector<std::uint16_t>(
        static_cast<std::size_t>(rows) * tokens,
        ninfer::test::f32_to_bf16(1.0F / static_cast<float>(rows)));
}

int compare_exact(const ninfer::DeviceBuffer& expected, const ninfer::DeviceBuffer& actual,
                  std::string_view label) {
    const std::vector<std::uint16_t> expected_bits =
        ninfer::test::from_device<std::uint16_t>(expected, expected.bytes / 2);
    const std::vector<std::uint16_t> actual_bits =
        ninfer::test::from_device<std::uint16_t>(actual, actual.bytes / 2);
    if (expected_bits == actual_bits) { return 0; }
    const auto mismatch = std::mismatch(expected_bits.begin(), expected_bits.end(),
                                        actual_bits.begin());
    std::cerr << label << ": compact shard differs at output element "
              << (mismatch.first - expected_bits.begin()) << '\n';
    return 1;
}

void launch_q4(const Tensor& input, const Weight& weight, std::int32_t row_begin, Tensor& output) {
    if (input.ne[1] == 1) {
        ninfer::ops::detail::q4_linear_swiglu_gemv_pair_shard_launch(
            input, weight, row_begin, output, nullptr);
    } else if (input.ne[1] <= 32) {
        ninfer::ops::detail::q4_linear_swiglu_small_t_shard_launch(
            input, weight, row_begin, output, nullptr);
    } else {
        ninfer::ops::detail::q4_linear_swiglu_mma_shard_launch(
            input, weight, row_begin, output, nullptr);
    }
}

void launch_q5(const Tensor& input, const Weight& weight, std::int32_t column_begin,
               Tensor& output) {
    if (input.ne[1] == 1) {
        ninfer::ops::detail::q5_linear_graph_partial_gemv_launch(
            input, weight, column_begin, output, nullptr);
    } else {
        ninfer::ops::detail::q5_linear_graph_partial_mma_launch(
            input, weight, column_begin, output, nullptr);
    }
}

int verify_q4_shard(std::int32_t row_begin, std::int32_t rows,
                    const PackedRows& full_weight) {
    constexpr std::array<std::int32_t, 4> kTokens = {1, 2, 33, 129};
    const PackedRows compact_weight = make_q4_gate_up(rows, row_begin);
    ninfer::DeviceBuffer full_device(full_weight.bytes.size());
    full_device.copy_from_host(full_weight.bytes.data(), full_weight.bytes.size());
    ninfer::DeviceBuffer compact_device(compact_weight.bytes.size());
    compact_device.copy_from_host(compact_weight.bytes.data(), compact_weight.bytes.size());
    const Weight full    = full_weight.device_weight(full_device.p);
    const Weight compact = compact_weight.device_weight(compact_device.p);

    int failures = 0;
    for (const std::int32_t tokens : kTokens) {
        const std::vector<std::uint16_t> activation = constant_activation(5120, tokens);
        ninfer::DeviceBuffer input(activation.size() * sizeof(std::uint16_t));
        input.copy_from_host(activation.data(), input.bytes);
        const std::size_t output_bytes =
            static_cast<std::size_t>(rows) * tokens * sizeof(std::uint16_t);
        ninfer::DeviceBuffer expected(output_bytes);
        ninfer::DeviceBuffer actual(output_bytes);
        Tensor x(input.p, DType::BF16, {5120, tokens});
        Tensor expected_tensor(expected.p, DType::BF16, {rows, tokens});
        Tensor actual_tensor(actual.p, DType::BF16, {rows, tokens});
        launch_q4(x, full, row_begin, expected_tensor);
        launch_q4(x, compact, 0, actual_tensor);
        ninfer::test::cuda_synchronize();
        failures += compare_exact(expected, actual,
                                  "Q4 graph shard T=" + std::to_string(tokens));
    }
    return failures;
}

int verify_q5_shard(std::int32_t column_begin, std::int32_t columns,
                    const PackedRows& full_weight) {
    constexpr std::array<std::int32_t, 3> kTokens = {1, 2, 129};
    const PackedRows compact_weight = make_q5_down(columns, column_begin / 64);
    ninfer::DeviceBuffer full_device(full_weight.bytes.size());
    full_device.copy_from_host(full_weight.bytes.data(), full_weight.bytes.size());
    ninfer::DeviceBuffer compact_device(compact_weight.bytes.size());
    compact_device.copy_from_host(compact_weight.bytes.data(), compact_weight.bytes.size());
    const Weight full    = full_weight.device_weight(full_device.p);
    const Weight compact = compact_weight.device_weight(compact_device.p);

    int failures = 0;
    for (const std::int32_t tokens : kTokens) {
        const std::vector<std::uint16_t> activation = constant_activation(columns, tokens);
        ninfer::DeviceBuffer input(activation.size() * sizeof(std::uint16_t));
        input.copy_from_host(activation.data(), input.bytes);
        const std::size_t output_bytes = 5120ULL * tokens * sizeof(std::uint16_t);
        ninfer::DeviceBuffer expected(output_bytes);
        ninfer::DeviceBuffer actual(output_bytes);
        Tensor x(input.p, DType::BF16, {columns, tokens});
        Tensor expected_tensor(expected.p, DType::BF16, {5120, tokens});
        Tensor actual_tensor(actual.p, DType::BF16, {5120, tokens});
        launch_q5(x, full, column_begin, expected_tensor);
        launch_q5(x, compact, 0, actual_tensor);
        ninfer::test::cuda_synchronize();
        failures += compare_exact(expected, actual,
                                  "Q5 graph shard T=" + std::to_string(tokens));
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const PackedRows full_q4 = make_q4_gate_up(17408, 0);
        int failures = verify_q4_shard(0, 8192, full_q4);
        failures += verify_q4_shard(8192, 9216, full_q4);

        // 3-way: 6144 / 6144 / 5120
        failures += verify_q4_shard(0, 6144, full_q4);
        failures += verify_q4_shard(6144, 6144, full_q4);
        failures += verify_q4_shard(12288, 5120, full_q4);

        const PackedRows full_q5 = make_q5_down(17408, 0);
        failures += verify_q5_shard(0, 8192, full_q5);
        failures += verify_q5_shard(8192, 9216, full_q5);

        // 3-way: 6144 / 6144 / 5120
        failures += verify_q5_shard(0, 6144, full_q5);
        failures += verify_q5_shard(6144, 6144, full_q5);
        failures += verify_q5_shard(12288, 5120, full_q5);
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " compact graph-shard kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "compact graph-shard kernel test failed: " << error.what() << '\n';
        return 1;
    }
}
