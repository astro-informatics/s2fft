#ifndef _S2FFT_CALLBACKS_CUH_
#define _S2FFT_CALLBACKS_CUH_

#include "cufft.h"
#include <iostream>
#include "hresult.h"
#include <cufftXt.h>
#include <cstddef>
#include <cuda_runtime.h>

#define CUFFT_CB_ST(isDouble) (isDouble ? CUFFT_CB_LD_COMPLEX_DOUBLE : CUFFT_CB_LD_COMPLEX)
#define CUFFT_CB_LD(isDouble) (isDouble ? CUFFT_CB_ST_COMPLEX_DOUBLE : CUFFT_CB_ST_COMPLEX)

typedef long long int int64;

namespace s2fftKernels {
enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

// Fundamental Functions

static __device__ int64 fft_shift(size_t offset, int64 *params) {
    int64 n = params[0];
    int64 dist = params[1];
    int64 first_element_offset = offset < dist ? 0 : dist;

    int64 half = n / 2;
    int64 normalized_offset = offset - first_element_offset;
    int64 shifted_index = normalized_offset + half;
    int64 indx = (shifted_index % n) + first_element_offset;

    return indx;
}

static __device__ int64 fft_shift_eq(size_t offset, int64 *params) {
    int64 n = params[0];
    int64 first_element_offset = (offset / n) * n;
    int64 offset_in_ring = first_element_offset + offset % n;

    int64 half = n / 2;
    int64 shifted_index = offset_in_ring + half;
    int64 indx = (shifted_index % n) + first_element_offset;

    return indx;
}

template <typename Complex>
static __device__ void normalize(Complex *element, int64 size) {
    float norm_factor = 1.0f / (float)size;
    element->x *= norm_factor;
    element->y *= norm_factor;
}

template <typename Complex>
static __device__ void normalize_ortho(Complex *element, int64 size) {
    float norm_factor = 1.0f / sqrtf((float)size);
    element->x *= norm_factor;
    element->y *= norm_factor;
}

// Callbacks

template <typename Complex>
static __device__ void fft_shift_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                    void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    int64 indx = fft_shift(offset, params);
    data[indx] = element;
}

template <typename Complex>
static __device__ void fft_shift_eq_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                       void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    int64 indx = fft_shift_eq(offset, params);
    data[indx] = element;
}

template <typename Complex>
static __device__ void fft_norm_ortho_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                         void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize_ortho(&element, params[0]);

    data[offset] = element;
}

template <typename Complex>
static __device__ void fft_norm_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                   void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize(&element, params[0]);

    data[offset] = element;
}

// Declare the callbacks with shifts
template <typename Complex>
static __device__ void fft_norm_ortho_shift_cb(void *dataOut, size_t offset, Complex element,
                                               void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize_ortho(&element, params[0]);
    int64 indx = fft_shift(offset, params);

    data[indx] = element;
}

template <typename Complex>
static __device__ void fft_norm_shift_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                         void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize(&element, params[0]);
    int64 indx = fft_shift(offset, params);

    data[indx] = element;
}

template <typename Complex>
static __device__ void fft_norm_ortho_shift_eq_cb(void *dataOut, size_t offset, Complex element,
                                                  void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize_ortho(&element, params[0]);
    int64 indx = fft_shift_eq(offset, params);

    data[indx] = element;
}

template <typename Complex>
static __device__ void fft_norm_shift_eq_cb(void *dataOut, size_t offset, Complex element, void *callerInfo,
                                            void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataOut;

    normalize(&element, params[0]);
    int64 indx = fft_shift_eq(offset, params);

    data[indx] = element;
}

// Inverse shifts

// Load callback because it has to be done before the inverse fft
template <typename Complex>
static __device__ Complex ifft_shift_cb(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataIn;
    int64 n = params[0];
    int64 dist = params[1];
    int64 first_element_offset = offset < dist ? 0 : dist;
    int64 normalized_offset = offset - first_element_offset;

    int64 half = n / 2;
    int64 shifted_index = normalized_offset - half;
    // Make sure that python % and C % are the same
    shifted_index = shifted_index < 0 ? n + shifted_index : shifted_index;
    int64 indx = (shifted_index % n) + first_element_offset;

    return data[indx];
}

template <typename Complex>
static __device__ Complex ifft_shift_eq_cb(void *dataIn, size_t offset, void *callerInfo,
                                           void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    Complex *data = (Complex *)dataIn;
    int64 n = params[0];
    int64 first_element_offset = (offset / n) * n;
    int64 offset_in_ring = first_element_offset + offset % n;

    int64 half = n / 2;
    int64 shifted_index = offset_in_ring - half;
    shifted_index = shifted_index < 0 ? n + shifted_index : shifted_index;
    int64 indx = (shifted_index % n) + first_element_offset;

    return data[indx];
}

// Ortho double
static __device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_no_shift_ptr =
        fft_norm_ortho_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_shift_ptr =
        fft_norm_ortho_shift_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackStoreZ fft_norm_ortho_double_shift_eq_ptr =
        fft_norm_ortho_shift_eq_cb<cufftDoubleComplex>;
// Ortho float
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_no_shift_ptr =
        fft_norm_ortho_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_shift_ptr =
        fft_norm_ortho_shift_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_float_shift_eq_ptr =
        fft_norm_ortho_shift_eq_cb<cufftComplex>;

// Norm FWD and BWD double
static __device__ __managed__ cufftCallbackStoreZ fft_norm_noshift_ptr = fft_norm_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackStoreZ fft_norm_shift_ptr = fft_norm_shift_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackStoreZ fft_norm_shift_eq_ptr =
        fft_norm_shift_eq_cb<cufftDoubleComplex>;
// Norm FWD and BWD float
static __device__ __managed__ cufftCallbackStoreC fft_norm_noshift_float_ptr = fft_norm_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackStoreC fft_norm_shift_float_ptr = fft_norm_shift_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackStoreC fft_norm_shift_eq_float_ptr =
        fft_norm_shift_eq_cb<cufftComplex>;

// Shifts double
static __device__ __managed__ cufftCallbackStoreZ fft_shift_double_ptr = fft_shift_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackStoreZ fft_shift_eq_double_ptr =
        fft_shift_eq_cb<cufftDoubleComplex>;
// Shifts float
static __device__ __managed__ cufftCallbackStoreC fft_shift_float_ptr = fft_shift_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackStoreC fft_shift_eq_float_ptr = fft_shift_eq_cb<cufftComplex>;
// ishift double
static __device__ __managed__ cufftCallbackLoadZ ifft_shift_double_ptr = ifft_shift_cb<cufftDoubleComplex>;
static __device__ __managed__ cufftCallbackLoadZ ifft_shift_eq_double_ptr =
        ifft_shift_eq_cb<cufftDoubleComplex>;
// ishift float
static __device__ __managed__ cufftCallbackLoadC ifft_shift_float_ptr = ifft_shift_cb<cufftComplex>;
static __device__ __managed__ cufftCallbackLoadC ifft_shift_eq_float_ptr = ifft_shift_eq_cb<cufftComplex>;

static __device__ __managed__ cufftCallbackStoreC fft_shift_eq_ptr = fft_shift_eq_cb;
static __device__ __managed__ cufftCallbackStoreC fft_shift_ptr = fft_shift_cb;
static __device__ __managed__ cufftCallbackLoadC ifft_shift_ptr = ifft_shift_cb;
static __device__ __managed__ cufftCallbackLoadC ifft_shift_eq_ptr = ifft_shift_eq_cb;

// This could have been done in a cleaner way perhaps.

static auto getIfftShiftDoubleCallback(bool equator) {
    if (equator) {
        return (void **)&ifft_shift_eq_double_ptr;
    } else {
        return (void **)&ifft_shift_double_ptr;
    }
}

static auto getIfftShiftFloatCallback(bool equator) {
    if (equator) {
        return (void **)&ifft_shift_eq_float_ptr;
    } else {
        return (void **)&ifft_shift_float_ptr;
    }
}

static auto getfftNormOrthoDouble(bool equator, bool shift) {
    if (equator) {
        if (shift) {
            return (void **)&fft_norm_ortho_double_shift_eq_ptr;
        } else {
            return (void **)&fft_norm_ortho_double_no_shift_ptr;
        }
    } else {
        if (shift) {
            return (void **)&fft_norm_ortho_double_shift_ptr;
        } else {
            return (void **)&fft_norm_ortho_double_no_shift_ptr;
        }
    }
}

static auto getfftNormOrthoFloat(bool equator, bool shift) {
    if (equator) {
        if (shift) {
            return (void **)&fft_norm_ortho_float_shift_eq_ptr;
        } else {
            return (void **)&fft_norm_ortho_float_no_shift_ptr;
        }
    } else {
        if (shift) {
            return (void **)&fft_norm_ortho_float_shift_ptr;
        } else {
            return (void **)&fft_norm_ortho_float_no_shift_ptr;
        }
    }
}

static auto getfftNormDouble(bool equator, bool shift) {
    if (equator) {
        if (shift) {
            return (void **)&fft_norm_shift_eq_ptr;
        } else {
            return (void **)&fft_norm_noshift_ptr;
        }
    } else {
        if (shift) {
            return (void **)&fft_norm_shift_ptr;
        } else {
            return (void **)&fft_norm_noshift_ptr;
        }
    }
}

static auto getfftNormFloat(bool equator, bool shift) {
    if (equator) {
        if (shift) {
            return (void **)&fft_norm_shift_eq_float_ptr;
        } else {
            return (void **)&fft_norm_noshift_float_ptr;
        }
    } else {
        if (shift) {
            return (void **)&fft_norm_shift_float_ptr;
        } else {
            return (void **)&fft_norm_noshift_float_ptr;
        }
    }
}

static auto getfftShiftDouble(bool equator) {
    if (equator) {
        return (void **)&fft_shift_eq_double_ptr;
    } else {
        return (void **)&fft_shift_double_ptr;
    }
}

static auto getfftShiftFloat(bool equator) {
    if (equator) {
        return (void **)&fft_shift_eq_float_ptr;
    } else {
        return (void **)&fft_shift_float_ptr;
    }
}

static HRESULT setCallback(cufftHandle forwardPlan, cufftHandle backwardPlan, int64 *params_dev, bool shift,
                           bool equator, bool doublePrecision, fft_norm norm) {
    //  Set the callback for the forward and backward
    if (shift) {
        if (doublePrecision) {
            CUFFT_CALL(cufftXtSetCallback(backwardPlan, getIfftShiftDoubleCallback(equator),
                                          CUFFT_CB_LD_COMPLEX_DOUBLE, (void **)&params_dev));
        } else {
            CUFFT_CALL(cufftXtSetCallback(backwardPlan, getIfftShiftFloatCallback(equator),
                                          CUFFT_CB_LD_COMPLEX, (void **)&params_dev));
        }
    }
    switch (norm) {
        case fft_norm::ORTHO:
            // ORTHO double shift
            // Shifting always happends in the load callback for the inverse fft
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftNormOrthoDouble(equator, shift),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
                CUFFT_CALL(cufftXtSetCallback(backwardPlan, getfftNormOrthoDouble(equator, false),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev))
                // ORTHO float shift
            } else {
                CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftNormOrthoFloat(equator, shift),
                                              CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
                CUFFT_CALL(cufftXtSetCallback(backwardPlan, getfftNormOrthoFloat(equator, false),
                                              CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
            }
            break;

        case fft_norm::BACKWARD:
            if (doublePrecision) {
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftShiftDouble(equator),
                                                  CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
                }
                CUFFT_CALL(cufftXtSetCallback(backwardPlan, getfftNormDouble(equator, false),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
            } else {
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftShiftFloat(equator), CUFFT_CB_ST_COMPLEX,
                                                  (void **)&params_dev));
                }
                CUFFT_CALL(cufftXtSetCallback(backwardPlan, getfftNormFloat(equator, false),
                                              CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
            }
            break;
        case fft_norm::FORWARD:
            if (doublePrecision) {
                CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftNormDouble(equator, shift),
                                              CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
            } else {
                CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftNormFloat(equator, shift),
                                              CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
            }
            break;
        case fft_norm::NONE:
            if (shift) {
                if (doublePrecision) {
                    CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftShiftDouble(equator),
                                                  CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)&params_dev));
                } else {
                    CUFFT_CALL(cufftXtSetCallback(forwardPlan, getfftShiftFloat(equator), CUFFT_CB_ST_COMPLEX,
                                                  (void **)&params_dev));
                }
            }
            break;
    }

    return S_OK;
}
}  // namespace s2fftKernels

#endif