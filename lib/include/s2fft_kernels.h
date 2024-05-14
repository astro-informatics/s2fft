#ifndef _S2FFT_KERNELS_H
#define _S2FFT_KERNELS_H

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include "cufft.h"
#include <cufftXt.h>
typedef long long int int64;

namespace s2fftKernels {

HRESULT launch_spectral_extension(cufftComplex* data, cufftComplex* output, const int& nside, const int& L,
                                  const int64& equatorial_offset_start,  const int64& equatorial_offset_end, cudaStream_t stream);
}

#endif  // _S2FFT_KERNELS_H