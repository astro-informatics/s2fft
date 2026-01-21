#include <iostream>
#include "hresult.h"
#include <cufft.h>
#include <cufftXt.h>
#include <cstddef>
#include <cuda_runtime.h>
#include "s2fft_callbacks.h"

namespace s2fftKernels {
// Fundamental Functions

/**
 * @brief Computes the shifted index for a 1D FFT.
 *
 * This function calculates the new index after applying an FFT shift,
 * which effectively moves the zero-frequency component to the center of the spectrum.
 *
 * @param offset The original offset (index) of the element.
 * @param params A pointer to an array containing FFT parameters: params[0] is n (size of FFT), params[1] is
 * dist (distance between batches).
 * @return The shifted index.
 */
__device__ int64 fft_shift(size_t offset, int64 *params) {
    // Step 1: Extract FFT size and distance between batches from parameters.
    int64 n = params[0];
    int64 dist = params[1];
    // Step 2: Determine the offset of the first element in the current batch.
    int64 first_element_offset = offset < dist ? 0 : dist;

    // Step 3: Calculate half the FFT size for shifting.
    int64 half = n / 2;
    // Step 4: Normalize the offset relative to the start of its batch.
    int64 normalized_offset = offset - first_element_offset;
    // Step 5: Apply the FFT shift.
    int64 shifted_index = normalized_offset + half;
    // Step 6: Calculate the final index, ensuring it wraps around correctly within the batch.
    int64 indx = (shifted_index % n) + first_element_offset;

    return indx;
}

/**
 * @brief Computes the shifted index for an equatorial FFT.
 *
 * This function calculates the new index after applying an FFT shift specifically
 * for the equatorial ring, where the data layout might differ slightly.
 *
 * @param offset The original offset (index) of the element.
 * @param params A pointer to an array containing FFT parameters: params[0] is n (size of FFT).
 * @return The shifted index.
 */
__device__ int64 fft_shift_eq(size_t offset, int64 *params) {
    // Step 1: Extract FFT size from parameters.
    int64 n = params[0];
    // Step 2: Calculate the starting offset of the current ring.
    int64 first_element_offset = (offset / n) * n;
    // Step 3: Calculate the offset within the current ring.
    int64 offset_in_ring = first_element_offset + offset % n;

    // Step 4: Calculate half the FFT size for shifting.
    int64 half = n / 2;
    // Step 5: Apply the FFT shift within the ring.
    int64 shifted_index = offset_in_ring + half;
    // Step 6: Calculate the final index, ensuring it wraps around correctly within the ring.
    int64 indx = (shifted_index % n) + first_element_offset;

    return indx;
}

/**
 * @brief Normalizes a complex element by dividing by the FFT size.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param element Pointer to the complex element to normalize.
 * @param size The size of the FFT.
 */
template <typename Complex>
__device__ void normalize(Complex *element, int64 size) {
    // Step 1: Calculate the normalization factor.
    float norm_factor = 1.0f / (float)size;
    // Step 2: Apply the normalization factor to the real part.
    element->x *= norm_factor;
    // Step 3: Apply the normalization factor to the imaginary part.
    element->y *= norm_factor;
}

/**
 * @brief Normalizes a complex element by dividing by the square root of the FFT size (orthonormalization).
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param element Pointer to the complex element to normalize.
 * @param size The size of the FFT.
 */
template <typename Complex>
__device__ void normalize_ortho(Complex *element, int64 size) {
    // Step 1: Calculate the orthonormalization factor.
    float norm_factor = 1.0f / sqrtf((float)size);
    // Step 2: Apply the normalization factor to the real part.
    element->x *= norm_factor;
    // Step 3: Apply the normalization factor to the imaginary part.
    element->y *= norm_factor;
}

// Callbacks

/**
 * @brief cuFFT callback function for applying FFT shift.
 *
 * This callback is executed by cuFFT to apply a circular shift to the output data.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_shift_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                             void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Calculate the shifted index.
    int64 indx = fft_shift(offset, params);
    // Step 4: Store the element at the shifted index.
    data[indx] = element;
}

/**
 * @brief cuFFT callback function for applying FFT shift to equatorial data.
 *
 * This callback is executed by cuFFT to apply a circular shift to the output data
 * specifically for the equatorial ring.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_shift_eq_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Calculate the shifted index for the equatorial ring.
    int64 indx = fft_shift_eq(offset, params);
    // Step 4: Store the element at the shifted index.
    data[indx] = element;
}

/**
 * @brief cuFFT callback function for applying orthonormalization.
 *
 * This callback is executed by cuFFT to normalize the output data by 1/sqrt(N).
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_ortho_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                  void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using orthonormalization.
    normalize_ortho(&element, params[0]);

    // Step 4: Store the normalized element at the original offset.
    data[offset] = element;
}

/**
 * @brief cuFFT callback function for applying standard normalization (1/N).
 *
 * This callback is executed by cuFFT to normalize the output data by 1/N.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                            void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using standard normalization.
    normalize(&element, params[0]);

    // Step 4: Store the normalized element at the original offset.
    data[offset] = element;
}

// Declare the callbacks with shifts
/**
 * @brief cuFFT callback function for applying orthonormalization and FFT shift.
 *
 * This callback combines orthonormalization and circular shifting of the output data.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_ortho_shift_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                        void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using orthonormalization.
    normalize_ortho(&element, params[0]);
    // Step 4: Calculate the shifted index.
    int64 indx = fft_shift(offset, params);

    // Step 5: Store the normalized element at the shifted index.
    data[indx] = element;
}

/**
 * @brief cuFFT callback function for applying standard normalization (1/N) and FFT shift.
 *
 * This callback combines standard normalization and circular shifting of the output data.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_shift_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                  void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using standard normalization.
    normalize(&element, params[0]);
    // Step 4: Calculate the shifted index.
    int64 indx = fft_shift(offset, params);

    // Step 5: Store the normalized element at the shifted index.
    data[indx] = element;
}

/**
 * @brief cuFFT callback function for applying orthonormalization and equatorial FFT shift.
 *
 * This callback combines orthonormalization and circular shifting of the output data
 * specifically for the equatorial ring.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_ortho_shift_eq_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                           void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using orthonormalization.
    normalize_ortho(&element, params[0]);
    // Step 4: Calculate the shifted index for the equatorial ring.
    int64 indx = fft_shift_eq(offset, params);

    // Step 5: Store the normalized element at the shifted index.
    data[indx] = element;
}

/**
 * @brief cuFFT callback function for applying standard normalization (1/N) and equatorial FFT shift.
 *
 * This callback combines standard normalization and circular shifting of the output data
 * specifically for the equatorial ring.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param dataOut Pointer to the output data buffer.
 * @param offset The current offset (index) within the output buffer.
 * @param element The complex element at the current offset.
 * @param callerInfo Pointer to user-defined parameters (params array).
 * @param sharedPointer Pointer to shared memory (unused in this callback).
 */
template <typename Complex>
__device__ void fft_norm_shift_eq_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                     void *sharedPointer) {
    // Step 1: Cast callerInfo to the correct parameter type.
    int64 *params = (int64 *)callerInfo;
    // Step 2: Cast dataOut to the correct complex data type.
    Complex *data = (Complex *)dataOut;

    // Step 3: Normalize the element using standard normalization.
    normalize(&element, params[0]);
    // Step 4: Calculate the shifted index for the equatorial ring.
    int64 indx = fft_shift_eq(offset, params);

    // Step 5: Store the normalized element at the shifted index.
    data[indx] = element;
}

// Pointers to device-managed cuFFT callback functions for different normalization and shift combinations.
// These are __managed__ to allow access from both host and device code.

// Ortho double precision callbacks
__device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_no_shift_ptr = fft_norm_ortho_cb;
__device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_shift_ptr = fft_norm_ortho_shift_cb;
__device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_shift_eq_ptr = fft_norm_ortho_shift_eq_cb;

// Ortho single precision callbacks
__device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_no_shift_ptr = fft_norm_ortho_cb;
__device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_shift_ptr = fft_norm_ortho_shift_cb;
__device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_shift_eq_ptr = fft_norm_ortho_shift_eq_cb;

// Standard (1/N) normalization double precision callbacks
__device__ __managed__ cufftCallbackStoreZ fft_norm_noshift_double_ptr = fft_norm_cb;
__device__ __managed__ cufftCallbackStoreZ fft_norm_shift_double_ptr = fft_norm_shift_cb;
__device__ __managed__ cufftCallbackStoreZ fft_norm_shift_eq_double_ptr = fft_norm_shift_eq_cb;

// Standard (1/N) normalization single precision callbacks
__device__ __managed__ cufftCallbackStoreC fft_norm_noshift_float_ptr = fft_norm_cb;
__device__ __managed__ cufftCallbackStoreC fft_norm_shift_float_ptr = fft_norm_shift_cb;
__device__ __managed__ cufftCallbackStoreC fft_norm_shift_eq_float_ptr = fft_norm_shift_eq_cb;

// Shift-only double precision callbacks
__device__ __managed__ cufftCallbackStoreZ fft_shift_double_ptr = fft_shift_cb;
__device__ __managed__ cufftCallbackStoreZ fft_shift_eq_double_ptr = fft_shift_eq_cb;

// Shift-only single precision callbacks
__device__ __managed__ cufftCallbackStoreC fft_shift_float_ptr = fft_shift_cb;
__device__ __managed__ cufftCallbackStoreC fft_shift_eq_float_ptr = fft_shift_eq_cb;

/**
 * @brief Returns the appropriate orthonormalization callback function pointer for double precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftNormOrthoDouble(bool equator, bool shift) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        // Step 1a: If equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_ortho_double_shift_eq_ptr;
        } else {
            return (void **)&fft_norm_ortho_double_no_shift_ptr;
        }
    } else {  // Step 1b: If not equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_ortho_double_shift_ptr;
        } else {
            return (void **)&fft_norm_ortho_double_no_shift_ptr;
        }
    }
}

/**
 * @brief Returns the appropriate orthonormalization callback function pointer for single precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftNormOrthoFloat(bool equator, bool shift) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        // Step 1a: If equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_ortho_float_shift_eq_ptr;
        } else {
            return (void **)&fft_norm_ortho_float_no_shift_ptr;
        }
    } else {  // Step 1b: If not equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_ortho_float_shift_ptr;
        } else {
            return (void **)&fft_norm_ortho_float_no_shift_ptr;
        }
    }
}

/**
 * @brief Returns the appropriate standard normalization callback function pointer for double precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftNormDouble(bool equator, bool shift) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        // Step 1a: If equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_shift_eq_double_ptr;
        } else {
            return (void **)&fft_norm_noshift_double_ptr;
        }
    } else {  // Step 1b: If not equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_shift_double_ptr;
        } else {
            return (void **)&fft_norm_noshift_double_ptr;
        }
    }
}

/**
 * @brief Returns the appropriate standard normalization callback function pointer for single precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @param shift Boolean flag indicating whether to apply FFT shifting.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftNormFloat(bool equator, bool shift) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        // Step 1a: If equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_shift_eq_float_ptr;
        } else {
            return (void **)&fft_norm_noshift_float_ptr;
        }
    } else {  // Step 1b: If not equatorial, check for shift.
        if (shift) {
            return (void **)&fft_norm_shift_float_ptr;
        } else {
            return (void **)&fft_norm_noshift_float_ptr;
        }
    }
}

/**
 * @brief Returns the appropriate shift-only callback function pointer for double precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftShiftDouble(bool equator) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        return (void **)&fft_shift_eq_double_ptr;
    } else {  // Step 1a: If not equatorial.
        return (void **)&fft_shift_double_ptr;
    }
}

/**
 * @brief Returns the appropriate shift-only callback function pointer for single precision.
 *
 * @param equator Boolean flag indicating if the current operation is for the equatorial ring.
 * @return A void** pointer to the selected cuFFT callback function.
 */
static auto getfftShiftFloat(bool equator) {
    // Step 1: Check if it's an equatorial ring.
    if (equator) {
        return (void **)&fft_shift_eq_float_ptr;
    } else {  // Step 1a: If not equatorial.
        return (void **)&fft_shift_float_ptr;
    }
}

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
                           bool doublePrecision, fft_norm norm) {
    // Step 1: Set the callback for the forward plan based on normalization type.
    switch (norm) {
        case fft_norm::ORTHO:
            // Step 1a: Orthonormalization with optional shift.
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormOrthoDouble(equator, shift),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
            } else {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormOrthoFloat(equator, shift), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
            }
            break;

        case fft_norm::BACKWARD:
            // Step 1b: Backward normalization. Apply shift only if requested.
            if (doublePrecision) {
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(plan, getfftShiftDouble(equator),
                                                  CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
                }
            } else {
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(plan, getfftShiftFloat(equator), CUFFT_CB_ST_COMPLEX,
                                                  (void **)&params_dev));
                }
            }
            break;
        case fft_norm::FORWARD:
            // Step 1c: Forward normalization. Apply normalization and shift.
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormDouble(equator, shift),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
            } else {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormFloat(equator, shift), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
            }
            break;
        case fft_norm::NONE:
            // Step 1d: No normalization. Apply shift only if requested.
            if (shift) {
                if (doublePrecision) {
                    CUFFT_CALL(cufftXtSetCallback(plan, getfftShiftDouble(equator),
                                                  CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
                } else {
                    CUFFT_CALL(cufftXtSetCallback(plan, getfftShiftFloat(equator), CUFFT_CB_ST_COMPLEX,
                                                  (void **)&params_dev));
                }
            }
            break;
    }

    return S_OK;
}

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
                            bool doublePrecision, fft_norm norm) {
    // Step 1: Set the callback for the backward plan based on normalization type.
    switch (norm) {
        case fft_norm::ORTHO:
            // Step 1a: Orthonormalization without shift (shift is handled in forward for ORTHO).
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormOrthoDouble(equator, false),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev))
            } else {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormOrthoFloat(equator, false), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
            }
            break;

        case fft_norm::BACKWARD:
            // Step 1b: Backward normalization without shift.
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormDouble(equator, false),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
            } else {
                CUFFT_CALL(cufftXtSetCallback(plan, getfftNormFloat(equator, false), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
            }
            break;
        case fft_norm::FORWARD:
        case fft_norm::NONE:
            // Step 1c: No normalization or forward normalization for backward plan.
            // No callback is set for these cases in the backward plan.
            break;
    }

    return S_OK;
}
}  // namespace s2fftKernels