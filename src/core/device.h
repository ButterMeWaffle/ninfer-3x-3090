#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// CUDA function attributes are scoped to a device context. A process-wide static
// result therefore leaves the same kernel unconfigured when it is first launched
// on a second GPU. Give each launcher specialization a cheap, device-keyed cache.
template <typename Configure>
void configure_cuda_device_once(Configure&& configure) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaError_t> results;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaError_t result = cudaSuccess;
    {
        const std::scoped_lock lock(mutex);
        const auto existing = results.find(device);
        if (existing != results.end()) {
            result = existing->second;
        } else {
            result = std::forward<Configure>(configure)();
            results.emplace(device, result);
        }
    }
    CUDA_CHECK(result);
}

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    explicit DeviceContext(std::span<const int> device_ids);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool model_parallel() const noexcept;
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept;
    [[nodiscard]] cudaStream_t stream_for_rank(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t fence_for_rank(std::size_t rank) const;
    void activate_rank(std::size_t rank);
    void synchronize_rank(std::size_t rank) const;
    void synchronize() const;

private:
    struct Endpoint {
        int device               = 0;
        cudaStream_t stream      = nullptr;
        cudaStream_t load_stream = nullptr;
        cudaEvent_t fence        = nullptr;
        cudaDeviceProp props{};
    };

    void refresh_active_aliases() noexcept;
    void release() noexcept;

    std::vector<Endpoint> endpoints_;
    std::vector<int> device_ids_;
    std::size_t active_rank_ = 0;
};

class ScopedDeviceRank {
public:
    ScopedDeviceRank(DeviceContext& context, std::size_t rank);
    ~ScopedDeviceRank() noexcept;

    ScopedDeviceRank(const ScopedDeviceRank&)            = delete;
    ScopedDeviceRank& operator=(const ScopedDeviceRank&) = delete;

private:
    DeviceContext& context_;
    std::size_t previous_rank_ = 0;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
