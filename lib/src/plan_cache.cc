#include "plan_cache.h"
#include <iostream>
#include <vector>
#include "s2fft.h"
#include "hresult.h"
#include <unordered_map>

namespace s2fft {

PlanCache::PlanCache() { is_initialized = true; }

HRESULT PlanCache::GetS2FFTExec(s2fftDescriptor &descriptor,
                                std::shared_ptr<s2fftExec<cufftComplex>> &executor) {
    HRESULT hr(E_FAIL);

    auto it = m_Descriptors32.find(descriptor);
    if (it != m_Descriptors32.end()) {
        executor = it->second;
        hr = S_FALSE;
    }

    if (hr == E_FAIL) {
        size_t worksize(0);
        hr = executor->Initialize(descriptor, worksize);
        if (SUCCEEDED(hr)) {
            m_Descriptors32[descriptor] = executor;
        }
    }
    return hr;
}

HRESULT PlanCache::GetS2FFTExec(s2fftDescriptor &descriptor,
                                std::shared_ptr<s2fftExec<cufftDoubleComplex>> &executor) {
    HRESULT hr(E_FAIL);

    auto it = m_Descriptors64.find(descriptor);
    if (it != m_Descriptors64.end()) {
        executor = it->second;
        hr = S_FALSE;
    }

    if (hr == E_FAIL) {
        size_t worksize(0);
        hr = executor->Initialize(descriptor, worksize);
        if (SUCCEEDED(hr)) {
            m_Descriptors64[descriptor] = executor;
        }
    }
    return hr;
}

}  // namespace s2fft
