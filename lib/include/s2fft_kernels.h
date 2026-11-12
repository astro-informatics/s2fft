#ifndef _S2FFT_KERNELS_H
#define _S2FFT_KERNELS_H

#include "cuda_check.h"
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
 * for better performance and maintainability.
 */

namespace s2fftKernels {

enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

/**
 * @brief Launches the spectral folding CUDA kernel.
 *
 * This function configures and launches the spectral_folding kernel with
 * appropriate grid and block dimensions. It performs spectral folding operations
 * on ring-ordered data, transforming from Fourier coefficient space to HEALPix
 * pixel space with optional FFT shifting and normalization.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input data array containing Fourier coefficients per ring.
 * @param output Output array for folded HEALPix pixel data.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param apply_shift Flag indicating whether to apply FFT shifting.
 * @param norm Normalization type (0=by nphi, 1=by sqrt(nphi), 2=no normalization).
 * @param stream CUDA stream for kernel execution.
 * @throws std::runtime_error if the launch fails.
 */
template <typename complex>
void launch_spectral_folding(complex* data, complex* output, const int& nside, const int& L,
                                const bool& apply_shift, const int& norm, cudaStream_t stream);

/**
 * @brief Launches the spectral extension CUDA kernel.
 *
 * This function configures and launches the spectral_extension kernel with
 * appropriate grid and block dimensions. It performs the inverse operation of
 * spectral folding, extending HEALPix pixel data back to full Fourier coefficient
 * space with optional FFT shifting and normalization.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input array containing folded HEALPix pixel data.
 * @param output Output array for extended Fourier coefficients per ring.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param apply_shift Flag indicating whether to apply FFT shifting.
 * @param norm Normalization type (0=by nphi, 1=by sqrt(nphi), 2=no normalization).
 * @param stream CUDA stream for kernel execution.
 * @throws std::runtime_error if the launch fails.
 */
template <typename complex>
void launch_spectral_extension(complex* data, complex* output, const int& nside, const int& L,
                                  const bool& apply_shift, const int& norm, cudaStream_t stream);

}  // namespace s2fftKernels

#endif  // _S2FFT_KERNELS_H