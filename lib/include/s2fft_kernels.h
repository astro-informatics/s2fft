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

enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

template <typename complex>
HRESULT launch_spectral_folding(complex* data, complex* output, const int& nside, const int& L,
                                const bool& shift, cudaStream_t stream);
template <typename complex>
HRESULT launch_spectral_extension(complex* data, complex* output, const int& nside, const int& L,
                                  cudaStream_t stream);

template <typename complex>
HRESULT launch_shift_normalize_kernel(cudaStream_t stream,
                                      complex* data,  // In-place data buffer
                                      int nside, bool apply_shift, int norm);

}  // namespace s2fftKernels

#endif  // _S2FFT_KERNELS_H