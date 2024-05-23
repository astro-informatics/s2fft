
#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda/std/complex>
#include "hresult.h"
#include "s2fft.h"
#include <unordered_map>

namespace s2fft {

class PlanCache {
public:
    static PlanCache &GetInstance() {
        static PlanCache instance;
        return instance;
    }

    HRESULT GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<cufftComplex>> &executor);

    HRESULT GetS2FFTExec(s2fftDescriptor &descriptor,
                         std::shared_ptr<s2fftExec<cufftDoubleComplex>> &executor);

    ~PlanCache() {}

private:
    bool is_initialized = false;

    std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<cufftDoubleComplex>>,
                       std::hash<s2fftDescriptor>, std::equal_to<>>
            m_Descriptors64;
    std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<cufftComplex>>, std::hash<s2fftDescriptor>,
                       std::equal_to<>>
            m_Descriptors32;

    PlanCache();

public:
    PlanCache(PlanCache const &) = delete;
    void operator=(PlanCache const &) = delete;
};
}  // namespace s2fft

#endif  // PLAN_CACHE_H