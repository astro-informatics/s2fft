#include "plan_cache.h"
#include <iostream>
#include <vector>
#include "s2fft.h"
#include "hresult.h"
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
 * @return HRESULT indicating success (S_OK if new, S_FALSE if from cache) or failure.
 */
template <typename T>
HRESULT PlanCache::GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<T>> &executor) {
    // Step 1: Check if the type is cufftComplex (single precision).
    if constexpr (std::is_same_v<T, cufftComplex>) {
        HRESULT hr(E_FAIL);
        // Step 1a: Try to find the descriptor in the single-precision cache.
        auto it = m_Descriptors32.find(descriptor);
        if (it != m_Descriptors32.end()) {
            // Step 1b: If found, retrieve the existing executor and set HR to S_FALSE (found in cache).
            executor = it->second;
            hr = S_FALSE;
        }

        // Step 1c: If not found (hr is still E_FAIL),
        if (hr == E_FAIL) {
            // Step 1d: Initialize a new executor with the descriptor.
            hr = executor->Initialize(descriptor);
            // Step 1e: If initialization is successful, store the new executor in the cache.
            if (SUCCEEDED(hr)) {
                m_Descriptors32[descriptor] = executor;
            }
        }
        // Step 1f: Return the HRESULT.
        return hr;
    } else {  // Step 2: If the type is not cufftComplex, it must be cufftDoubleComplex (double precision).
        HRESULT hr(E_FAIL);
        // Step 2a: Try to find the descriptor in the double-precision cache.
        auto it = m_Descriptors64.find(descriptor);
        if (it != m_Descriptors64.end()) {
            // Step 2b: If found, retrieve the existing executor and set HR to S_FALSE (found in cache).
            executor = it->second;
            hr = S_FALSE;
        }

        // Step 2c: If not found (hr is still E_FAIL),
        if (hr == E_FAIL) {
            // Step 2d: Initialize a new executor with the descriptor.
            hr = executor->Initialize(descriptor);
            // Step 2e: If initialization is successful, store the new executor in the cache.
            if (SUCCEEDED(hr)) {
                m_Descriptors64[descriptor] = executor;
            }
        }
        // Step 2f: Return the HRESULT.
        return hr;
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
template HRESULT PlanCache::GetS2FFTExec<cufftComplex>(s2fftDescriptor &descriptor,
                                                       std::shared_ptr<s2fftExec<cufftComplex>> &executor);

template HRESULT PlanCache::GetS2FFTExec<cufftDoubleComplex>(
        s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<cufftDoubleComplex>> &executor);

}  // namespace s2fft