#include "plan_cache.h"
#include <iostream>
#include <vector>
#include "s2fft.h"
#include <unordered_map>

namespace s2fft {

/**
 * @brief Constructor for PlanCache.
 *
 * Initializes the `is_initialized` flag to true.
 */
PlanCache::PlanCache() {
    // Step 1: Set the initialization flag.
    is_initialized = true;
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
 * @param device_ordinal The CUDA device the executor must belong to (see plan_cache.h).
 * @return true if the executor was created and initialised here, false if it came from the cache.
 * @throws std::runtime_error if initialising a new executor fails.
 */
template <typename T>
bool PlanCache::GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<T>> &executor,
                             int32_t device_ordinal) {
    // Step 1: Check if the type is cufftComplex (single precision).
    if constexpr (std::is_same_v<T, cufftComplex>) {
        // Step 1a: Try to find the descriptor in the device's single-precision cache.
        auto &cache = m_Descriptors32[device_ordinal];
        auto it = cache.find(descriptor);
        if (it != cache.end()) {
            // Step 1b: If found, return the cached executor.
            executor = it->second;
            return false;
        }

        // Step 1c: Otherwise,
        {
            // Step 1d: Initialize a new executor with the descriptor. The caller holds a device
            // guard, so the cuFFT plans created here bind to the requested device.
            executor->Initialize(descriptor);
            // Step 1e: Store the new executor in the cache (Initialize throws on failure).
            cache[descriptor] = executor;
        }
        // Step 1f: The executor is new.
        return true;
    } else {  // Step 2: If the type is not cufftComplex, it must be cufftDoubleComplex (double precision).
        // Step 2a: Try to find the descriptor in the device's double-precision cache.
        auto &cache = m_Descriptors64[device_ordinal];
        auto it = cache.find(descriptor);
        if (it != cache.end()) {
            // Step 2b: If found, return the cached executor.
            executor = it->second;
            return false;
        }

        // Step 2c: Otherwise,
        {
            // Step 2d: Initialize a new executor with the descriptor. The caller holds a device
            // guard, so the cuFFT plans created here bind to the requested device.
            executor->Initialize(descriptor);
            // Step 2e: Store the new executor in the cache (Initialize throws on failure).
            cache[descriptor] = executor;
        }
        // Step 2f: The executor is new.
        return true;
    }
}

/**
 * @brief Clears all cached s2fftExec instances.
 *
 * This method is typically called during application shutdown to release
 * all resources held by the cached FFT plans.
 */
void PlanCache::Finalize() {
    // Step 1: Check if the cache was initialized.
    if (is_initialized) {
        // Step 1a: Clear both single and double precision descriptor maps.
        m_Descriptors32.clear();
        m_Descriptors64.clear();
    }
    // Step 2: Reset the initialization flag.
    is_initialized = false;
}

/**
 * @brief Destructor for PlanCache.
 *
 * Ensures that Finalize() is called when the PlanCache instance is destroyed,
 * performing necessary cleanup.
 */
PlanCache::~PlanCache() {
    // Step 1: Call Finalize to clean up resources.
    Finalize();
}

// Explicitly instantiate the templates for the supported complex types.
// This is necessary for the linker to find the concrete implementations of the templated function.
template bool PlanCache::GetS2FFTExec<cufftComplex>(s2fftDescriptor &descriptor,
                                                    std::shared_ptr<s2fftExec<cufftComplex>> &executor,
                                                    int32_t device_ordinal);

template bool PlanCache::GetS2FFTExec<cufftDoubleComplex>(
        s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<cufftDoubleComplex>> &executor,
        int32_t device_ordinal);

}  // namespace s2fft