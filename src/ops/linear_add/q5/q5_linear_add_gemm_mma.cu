#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C16Schedule =
    Q5RowSplitMmaGemmSchedule<64, 16, 64, 16, 8, 2, 3, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C24Schedule =
    Q5RowSplitMmaGemmSchedule<64, 24, 64, 16, 8, 2, 2, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C64Schedule =
    Q5RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q5FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q5ScaleLoad::Scalar16>;
using MmaR64C128Schedule =
    Q5RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;

template <class Schedule, bool Full>
void launch_kernel(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* out                   = static_cast<__nv_bfloat16*>(residual_out.data);
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    q5_rowsplit_gemm_mma_kernel<Schedule, Full, Q5MmaEpilogue::CtaCollectiveResidual>
        <<<grid, Schedule::kThreads, 0, stream>>>(xp, codes, high, scales, out, out, rows, k, cols,
                                                  padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool full = (w.n % 64) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % 64) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice  = x.slice(1, offset, count);
                             Tensor residual_slice = residual_out.slice(1, offset, count);
                             if (full) {
                                 launch_kernel<Schedule, true>(x_slice, w, residual_slice, stream);
                             } else {
                                 launch_kernel<Schedule, false>(x_slice, w, residual_slice, stream);
                             }
                         });
}

template <class Schedule>
void launch_graph_partial(const Tensor& x, const Weight& w, std::int32_t input_row_begin,
                          Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kTile = Schedule::kBlockK;
    const std::int32_t cols      = x.ne[1];
    const std::int32_t tile_begin = input_row_begin / kTile;
    const std::int32_t tile_count = x.ne[0] / kTile;
    for_each_token_slice(cols, Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             const dim3 grid(
                                 static_cast<unsigned>(div_up(out.ne[0], Schedule::kBlockRows)),
                                 static_cast<unsigned>(div_up(count, Schedule::kBlockCols)), 1u);
                             q5_rowsplit_gemm_mma_kernel<Schedule, false, Q5MmaEpilogue::Store>
                                 <<<grid, Schedule::kThreads, 0, stream>>>(
                                     static_cast<const __nv_bfloat16*>(x_slice.data),
                                     static_cast<const std::uint8_t*>(w.qdata),
                                     static_cast<const std::uint8_t*>(w.qhigh),
                                     static_cast<const std::uint8_t*>(w.scales), nullptr,
                                     static_cast<__nv_bfloat16*>(out_slice.data), out.ne[0],
                                     x.ne[0], count, w.padded_shape[1], tile_begin, tile_count);
                             CUDA_CHECK(cudaGetLastError());
                         });
}

} // namespace

void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<MmaR64C16Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<MmaR64C24Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<MmaR64C64Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    launch_route<MmaR64C128Schedule>(x, w, residual_out, stream);
}

void q5_linear_graph_partial_mma_launch(const Tensor& x, const Weight& w,
                                        std::int32_t input_row_begin, Tensor& out,
                                        cudaStream_t stream) {
    constexpr std::int32_t kParentK = 17408;
    const bool full_parent = w.k == kParentK && w.padded_shape[1] == kParentK &&
                             input_row_begin + x.ne[0] <= kParentK;
    const bool compact = (w.k == 8192 || w.k == 9216) && w.padded_shape[1] == w.k &&
                         x.ne[0] == w.k && input_row_begin == 0;
    if (w.n != 5120 || (!full_parent && !compact) || x.ne[1] <= 1 ||
        out.ne[0] != 5120 || out.ne[1] != x.ne[1] || input_row_begin < 0 || x.ne[0] <= 0 ||
        (input_row_begin % 64) != 0 ||
        (x.ne[0] % 64) != 0) {
        throw std::invalid_argument("q5 graph-partial MMA has invalid geometry");
    }
    if (x.ne[1] <= 32) {
        launch_graph_partial<MmaR64C16Schedule>(x, w, input_row_begin, out, stream);
    } else if (x.ne[1] <= 48) {
        launch_graph_partial<MmaR64C24Schedule>(x, w, input_row_begin, out, stream);
    } else if (x.ne[1] <= 128) {
        launch_graph_partial<MmaR64C64Schedule>(x, w, input_row_begin, out, stream);
    } else {
        launch_graph_partial<MmaR64C128Schedule>(x, w, input_row_begin, out, stream);
    }
}

} // namespace ninfer::ops::detail
