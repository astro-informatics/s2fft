#ifndef _S2FFT_CALLBACKS_CUH_
#define _S2FFT_CALLBACKS_CUH_

#include <cufft.h>
#include <cufftXt.h>
#include <iostream>
#include "hresult.h"
#include <cufftXt.h>
#include <cstddef>
#include <cuda_runtime.h>

typedef long long int int64;

namespace s2fftKernels {
enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

HRESULT setCallback(cufftHandle forwardPlan, cufftHandle backwardPlan, int64 *params_dev, bool shift,
                    bool equator, bool doublePrecision, fft_norm norm);
}  // namespace s2fftKernels

#endif