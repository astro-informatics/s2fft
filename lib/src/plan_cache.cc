#include "plan_cache.h"
#include "logger.hpp"
#include <iostream>
#include <vector>
#include "s2fft.h"
#include "hresult.h"
#include <unordered_map>

namespace s2fft {

PlanCache::PlanCache() { is_initialized = true; }

}  // namespace s2fft