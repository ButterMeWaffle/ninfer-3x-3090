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

std::vector<int> find_peer_set(int count, int wanted) {
    std::vector<int> chosen;
    for (int id = 0; id < count && static_cast<int>(chosen.size()) < wanted; ++id) {
        if (std::find(chosen.begin(), chosen.end(), id) != chosen.end()) { continue; }
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, id));
        bool fits = true;
        for (const int peer : chosen) {
            cudaDeviceProp peer_properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&peer_properties, peer));
            int id_to_peer = 0;
            int peer_to_id = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&id_to_peer, id, peer));
            CUDA_CHECK(cudaDeviceCanAccessPeer(&peer_to_id, peer, id));
            if (id_to_peer == 0 || peer_to_id == 0 || properties.major != peer_properties.major ||
                properties.minor != peer_properties.minor) {
                fits = false;
                break;
            }
        }
        if (fits) { chosen.push_back(id); }
    }
    return chosen.size() == static_cast<std::size_t>(wanted) ? chosen
                                                             : std::vector<int>{};
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

    const std::vector<int> peer_pair = find_peer_set(count, 2);
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

    const std::vector<int> peer_triple = find_peer_set(count, 3);
    if (!peer_triple.empty()) {
        ninfer::DeviceContext triple{std::span<const int>(peer_triple)};
        if (!triple.model_parallel() || triple.size() != 3 ||
            triple.device_ids() != peer_triple || triple.active_rank() != 0 ||
            triple.fence_for_rank(0) == nullptr || triple.fence_for_rank(1) == nullptr ||
            triple.fence_for_rank(2) == nullptr) {
            ++failures;
            std::cerr << "three-device context topology is invalid\n";
        }
        triple.activate_rank(2);
        int active_device = -1;
        CUDA_CHECK(cudaGetDevice(&active_device));
        if (active_device != peer_triple[2] || triple.device != peer_triple[2]) {
            ++failures;
            std::cerr << "rank activation did not select the third device\n";
        }
        triple.activate_rank(0);
        triple.synchronize();
    }

    return failures == 0 ? 0 : fail("device test failed");
}
