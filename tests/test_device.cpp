#include "core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
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

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    const std::vector<int> peer_pair = find_peer_pair(count);
    if (!peer_pair.empty()) {
        ninfer::DeviceContext dual{std::span<const int>(peer_pair)};
        if (!dual.model_parallel() || dual.size() != 2 || dual.device_ids() != peer_pair ||
            dual.active_rank() != 0 || dual.fence_for_rank(0) == nullptr ||
            dual.fence_for_rank(1) == nullptr) {
            ++failures;
            std::cerr << "two-device context topology is invalid\n";
        }
        dual.activate_rank(1);
        int active_device = -1;
        CUDA_CHECK(cudaGetDevice(&active_device));
        if (active_device != peer_pair[1] || dual.device != peer_pair[1]) {
            ++failures;
            std::cerr << "rank activation did not select the secondary device\n";
        }
        dual.activate_rank(0);
        dual.synchronize();

        bool duplicate_rejected = false;
        try {
            const std::vector<int> duplicate = {peer_pair[0], peer_pair[0]};
            ninfer::DeviceContext invalid{std::span<const int>(duplicate)};
        } catch (const std::invalid_argument&) { duplicate_rejected = true; }
        if (!duplicate_rejected) {
            ++failures;
            std::cerr << "duplicate device ids were accepted\n";
        }
    }

    return failures == 0 ? 0 : fail("device test failed");
}
