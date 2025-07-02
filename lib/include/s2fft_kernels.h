#ifndef _S2FFT_KERNELS_H
#define _S2FFT_KERNELS_H

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include "cufft.h"
#include <cufftXt.h>
typedef long long int int64;

/**
 * @file s2fft_kernels.h
 * @brief CUDA kernels for HEALPix spherical harmonic transforms
 *
 * @note CUFT CALLBACKS DEPRECATED: This implementation no longer uses cuFFT callbacks.
 * The previous callback-based approach has been replaced with direct kernel launches
 * for better performance and maintainability. The files s2fft_callbacks.h and
 * s2fft_callbacks.cc are no longer used and can be considered orphaned.
 */

namespace s2fftKernels {

enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

/**
 * @brief Launches the spectral folding CUDA kernel.
 *
 * This function configures and launches the spectral_folding kernel with
 * appropriate grid and block dimensions. It performs spectral folding operations
 * on ring-ordered data, transforming from Fourier coefficient space to HEALPix
 * pixel space with optional FFT shifting.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input data array containing Fourier coefficients per ring.
 * @param output Output array for folded HEALPix pixel data.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param shift Flag indicating whether to apply FFT shifting.
 * @param stream CUDA stream for kernel execution.
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_spectral_folding(complex* data, complex* output, const int& nside, const int& L,
                                const bool& shift, cudaStream_t stream);

/**
 * @brief Launches the spectral extension CUDA kernel.
 *
 * This function configures and launches the spectral_extension kernel with
 * appropriate grid and block dimensions. It performs the inverse operation of
 * spectral folding, extending HEALPix pixel data back to full Fourier coefficient
 * space by mapping folded frequency components to their appropriate positions.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input array containing folded HEALPix pixel data.
 * @param output Output array for extended Fourier coefficients per ring.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param stream CUDA stream for kernel execution.
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_spectral_extension(complex* data, complex* output, const int& nside, const int& L,
                                  cudaStream_t stream);

/**
 * @brief Launches the shift/normalize CUDA kernel for HEALPix data processing.
 *
 * This function configures and launches the shift_normalize_kernel with appropriate
 * grid and block dimensions. It handles both single and double precision complex
 * types and applies the requested normalization and shifting operations to HEALPix
 * pixel data on a per-ring basis.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param stream CUDA stream for kernel execution.
 * @param data Input/output array of HEALPix pixel data (in-place processing).
 * @param nside The HEALPix Nside parameter.
 * @param apply_shift Flag indicating whether to apply FFT shifting.
 * @param norm Normalization type (0=by nphi, 1=by sqrt(nphi), 2=no normalization).
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_shift_normalize_kernel(cudaStream_t stream,
                                      complex* data,  // In-place data buffer
                                      int nside, bool apply_shift, int norm);

}  // namespace s2fftKernels

#endif  // _S2FFT_KERNELS_H