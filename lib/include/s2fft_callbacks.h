/**
 * @file s2fft_callbacks.h
 * @brief CUDA CUFFT callbacks for HEALPix spherical harmonic transforms
 * 
 * @note CUFFT CALLBACKS DEPRECATED: This implementation no longer uses cuFFT callbacks.
 * The previous callback-based approach has been replaced with direct kernel launches
 * for better performance and maintainability. The files s2fft_callbacks.h and 
 * s2fft_callbacks.cc are no longer used and can be considered orphaned.
 */


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
/**
 * @brief Defines the normalization types for FFT operations.
 */
enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

/**
 * @brief Sets cuFFT callbacks specifically for a forward FFT plan.
 *
 * This function configures the cuFFT library to use custom callbacks
 * for normalization and shifting operations during forward FFT execution.
 *
 * @param plan The cuFFT handle for the forward FFT plan.
 * @param params_dev Pointer to device memory containing parameters for the callbacks.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param doublePrecision Boolean flag indicating if double precision is used.
 * @param norm The FFT normalization type to apply.
 * @return HRESULT indicating success or failure.
 */
HRESULT setForwardCallback(cufftHandle plan, int64 *params_dev, bool shift, bool equator,
                           bool doublePrecision, fft_norm norm);

/**
 * @brief Sets cuFFT callbacks specifically for a backward FFT plan.
 *
 * This function configures the cuFFT library to use custom callbacks
 * for normalization and shifting operations during backward FFT execution.
 *
 * @param plan The cuFFT handle for the inverse FFT plan.
 * @param params_dev Pointer to device memory containing parameters for the callbacks.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param doublePrecision Boolean flag indicating if double precision is used.
 * @param norm The FFT normalization type to apply.
 * @return HRESULT indicating success or failure.
 */
HRESULT setBackwardCallback(cufftHandle plan, int64 *params_dev, bool shift, bool equator,
                            bool doublePrecision, fft_norm norm);
}  // namespace s2fftKernels

#endif