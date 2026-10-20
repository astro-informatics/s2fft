#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda/std/complex>
#include "hresult.h"
#include "s2fft.h"
#include <unordered_map>
#include <type_traits>

namespace s2fft {

/**
 * @brief Manages and caches s2fftExec instances to optimize resource usage.
 *
 * This class implements the singleton pattern to ensure only one instance
 * of the PlanCache exists throughout the application. It stores pre-initialized
 * s2fftExec objects based on their descriptors (parameters like nside, L, etc.)
 * to avoid redundant initialization, which can be computationally expensive.
 */
class PlanCache {
public:
    /**
     * @brief Returns the singleton instance of the PlanCache.
     *
     * @return A reference to the single PlanCache instance.
     */
    static PlanCache &GetInstance() {
        static PlanCache instance;
        return instance;
    }

    /**
     * @brief Retrieves an s2fftExec instance from the cache or initializes a new one.
     *
     * This templated method attempts to find an existing s2fftExec instance
     * matching the provided descriptor in its internal cache (m_Descriptors32 or m_Descriptors64)
     * based on the Complex type T. If a matching instance is found, it is returned.
     * Otherwise, a new s2fftExec instance is created, initialized with the descriptor,
     * and then stored in the cache before being returned.
     *
     * @tparam T The complex type (cufftComplex or cufftDoubleComplex) of the s2fftExec instance.
     * @param descriptor The s2fftDescriptor containing the parameters for the FFT.
     * @param executor A shared_ptr that will point to the retrieved or newly initialized s2fftExec instance.
     * @param device_ordinal The CUDA device the executor must belong to. cuFFT plans bind to the
     *        device that is current when they are created, so the cache is keyed per device
     *        (single-process multi-GPU, astro-informatics/s2fft#389). A negative ordinal is a
     *        host-side warm-up lookup (never used by the runtime paths).
     * @return HRESULT indicating success (S_OK if new, S_FALSE if from cache) or failure.
     */
    template <typename T>
    HRESULT GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<T>> &executor,
                         int32_t device_ordinal);

    /**
     * @brief Clears all cached s2fftExec instances.
     *
     * This method is typically called during application shutdown to release
     * all resources held by the cached FFT plans.
     */
    void Finalize();

    /**
     * @brief Destructor for PlanCache.
     *
     * Ensures that Finalize() is called when the PlanCache instance is destroyed,
     * performing necessary cleanup.
     */
    ~PlanCache();

private:
    bool is_initialized = false;

    // Per-device caches of s2fftExec instances for double and single precision. cuFFT plans
    // live in the CUDA context of the device they were created on, so a process-global,
    // descriptor-only cache would hand device-0 plans to kernels running on other devices.
    std::unordered_map<int32_t,
                       std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<cufftDoubleComplex>>,
                                          std::hash<s2fftDescriptor>, std::equal_to<>>>
            m_Descriptors64;
    std::unordered_map<int32_t, std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<cufftComplex>>,
                                                   std::hash<s2fftDescriptor>, std::equal_to<>>>
            m_Descriptors32;

    /**
     * @brief Private constructor for PlanCache.
     *
     * Initializes the PlanCache instance. This constructor is private to enforce
     * the singleton pattern.
     */
    PlanCache();

public:
    // Delete copy constructor and assignment operator to prevent copying
    PlanCache(PlanCache const &) = delete;
    void operator=(PlanCache const &) = delete;
};
}  // namespace s2fft

#endif  // PLAN_CACHE_H