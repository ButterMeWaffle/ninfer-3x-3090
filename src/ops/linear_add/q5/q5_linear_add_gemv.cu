#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 6144, 16, 2, true>(xp, codes, high, scales,
                                                                         out, stream);
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 17408, 16, 2, false>(xp, codes, high, scales,
                                                                           out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add GEMV: unsupported exact shape");
    }
    CUDA_CHECK(cudaGetLastError());
}

void q5_linear_graph_partial_gemv_launch(const Tensor& x, const Weight& w,
                                         std::int32_t input_row_begin, Tensor& out,
                                         cudaStream_t stream) {
    constexpr int kRows = 5120;
    constexpr int kK    = 17408;
    constexpr int kTile = 1024;
    const bool full_parent = w.k == kK && w.padded_shape[1] == kK &&
                             input_row_begin + x.ne[0] <= kK;
    const bool compact = (w.k == 8192 || w.k == 9216 || w.k == 6144 || w.k == 5120) &&
                         w.padded_shape[1] == w.k &&
                         x.ne[0] == w.k && input_row_begin == 0;
    if (w.n != kRows || (!full_parent && !compact) || x.ne[1] != 1 ||
        out.ne[0] != kRows || out.ne[1] != 1 || input_row_begin < 0 || x.ne[0] <= 0 ||
        (input_row_begin % kTile) != 0 ||
        (x.ne[0] % kTile) != 0) {
        throw std::invalid_argument("q5 graph-partial GEMV has invalid geometry");
    }
    const int tile_begin = input_row_begin / kTile;
    const int tile_count = x.ne[0] / kTile;
    const auto launch = [&]<int PackedK>() {
        q5_rowsplit_gemv_kernel<kRows, PackedK, 16, 2, false, false>
            <<<kRows / 16, 16 * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.qhigh),
                static_cast<const std::uint8_t*>(w.scales),
                static_cast<__nv_bfloat16*>(out.data), nullptr, Q5GemvStoreEpilogue{}, tile_begin,
                tile_count);
    };
    if (w.k == 8192) {
        launch.template operator()<8192>();
    } else if (w.k == 9216) {
        launch.template operator()<9216>();
    } else if (w.k == 6144) {
        launch.template operator()<6144>();
    } else if (w.k == 5120) {
        launch.template operator()<5120>();
    } else {
        launch.template operator()<kK>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
