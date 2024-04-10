
#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include "logger.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda/std/complex>
#include "hresult.h"

namespace s2fft {

class PlanCache {
public:
    static PlanCache &GetInstance() {
        static PlanCache instance;
        return instance;
    }

    HRESULT GetS2FFTExec();

    ~PlanCache() {}
private:
    bool is_initialized = false;

    PlanCache();
public:
    PlanCache(PlanCache const &) = delete;
    void operator=(PlanCache const &) = delete;
};
}  // namespace s2fft


#endif // PLAN_CACHE_H