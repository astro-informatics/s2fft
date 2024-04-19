#ifndef _S2FFT_CALLBACKS_CUH_
#define _S2FFT_CALLBACKS_CUH_

#include "cufft.h"
#include <cufftXt.h>
#include <cstddef>
#include <cuda_runtime.h>

typedef long long int int64;

#define FFT_NORM(shift) (shift ? s2fftKernels::fft_norm_shift_ptr : s2fftKernels::fft_norm_noshift_ptr)
#define FFT_NORM_ORTHO(shift) \
    (shift ? s2fftKernels::fft_norm_ortho_shift_ptr : s2fftKernels::fft_norm_ortho_noshift_ptr)

#define FFT_NORM_EQ(shift) (shift ? s2fftKernels::fft_norm_shift_eq_ptr : s2fftKernels::fft_norm_noshift_ptr)
#define FFT_NORM_ORTHO_EQ(shift) \
    (shift ? s2fftKernels::fft_norm_ortho_shift_eq_ptr : s2fftKernels::fft_norm_ortho_noshift_ptr)

namespace s2fftKernels {

// Fundamental Functions

static __device__ void fft_shift(cufftComplex *data, size_t offset, cufftComplex element, int64* params) {
    int64 n = params[0];
    int64 dist = params[1];
    int64 upper_ring_offset = params[2];
    int64 lower_ring_offset = params[3];
    int64 first_element_offset = offset < dist ? 0 : dist;

    double half_f = n / 2.0;
    int64 half = ceil(half_f);
    int64 normalized_offset = offset - first_element_offset;
    size_t indx = ((normalized_offset + half) % n) + first_element_offset;

    data[indx] = element;
}

static __device__ void fft_shift_eq(void *dataOut, size_t offset, cufftComplex element, int64* params) {
    cufftComplex *data = (cufftComplex *)dataOut;
    int64 n = params[0];
    int64 equator_offset = params[1];
    int64 first_element_offset = ((offset - equator_offset) / n) * n + equator_offset;

    double half_f = n / 2.0;
    int64 half = ceil(half_f);
    size_t indx = ((offset + half) % n) + first_element_offset;
    data[indx] = element;
}

static __device__ void normalize(cufftComplex* element, int64 size) {
    double norm_factor = 1.0 / (double)size;
    element->x *= norm_factor;
    element->y *= norm_factor;
}

static __device__ void normalize_ortho(cufftComplex* element, int64 size) {
    double norm_factor = 1.0 / sqrtf((double)size);
    element->x *= norm_factor;
    element->y *= norm_factor;
}

// Callbacks

static __device__ void fft_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                    void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    fft_shift(data, offset, element, params);
}

static __device__ void fft_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                       void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    fft_shift_eq(data, offset, element, params);
}

static __device__ void fft_norm_ortho_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                         void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize_ortho(&element, params[0]);

    data[offset] = element;
}

static __device__ void fft_norm_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                   void *sharedPointer) {

    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize(&element, params[0]);

    data[offset] = element;
}

// Declare the callbacks with shifts

static __device__ void fft_norm_ortho_shift_cb(void *dataOut, size_t offset, cufftComplex element,
                                               void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize_ortho(&element, params[0]);
    fft_shift(data, offset, element, params);
}

static __device__ void fft_norm_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                         void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize(&element, params[0]);
    fft_shift(data, offset, element, params);
}

static __device__ void fft_norm_ortho_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                                  void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;    
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize_ortho(&element, params[0]);
    fft_shift_eq(data, offset, element, params);
}

static __device__ void fft_norm_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                            void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize(&element, params[0]);
    fft_shift_eq(data, offset, element, params);
}

// Inverse shifts

// Load callback because it has to be done before the inverse fft
static __device__ cufftComplex ifft_shift_cb(void *dataIn, size_t offset, void *callerInfo,
                                             void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataIn;
    int64 n = params[0];
    int64 dist = params[1];
    int64 first_element_offset = offset < dist ? 0 : dist;
    int64 normalized_offset = offset - first_element_offset;

    double half_f = n / 2.0;
    int64 half = floor(half_f);
    size_t indx = ((normalized_offset + half) % n) + first_element_offset;

    return data[indx];
}

static __device__ cufftComplex ifft_shift_eq_cb(void *dataIn, size_t offset, void *callerInfo,
                                                void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataIn;
    int64 n = params[0];
    int64 equator_offset = params[1];
    int64 first_element_offset = ((offset - equator_offset) / n) * n + equator_offset;

    // printf("offset: %lld, equator_offset: %lld, first_element_offset: %lld\n", offset, equator_offset,
    //        first_element_offset);

    double half_f = n / 2.0;
    int64 half = floor(half_f);
    size_t indx = ((offset + half) % n) + first_element_offset;

    return data[indx];
}

// Ortho
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_noshift_ptr = fft_norm_ortho_cb;
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_shift_ptr = fft_norm_ortho_shift_cb;
static __device__ __managed__ cufftCallbackStoreC fft_norm_ortho_shift_eq_ptr = fft_norm_ortho_shift_eq_cb;
// Norm FWD and BWD
static __device__ __managed__ cufftCallbackStoreC fft_norm_noshift_ptr = fft_norm_cb;
static __device__ __managed__ cufftCallbackStoreC fft_norm_shift_ptr = fft_norm_shift_cb;
static __device__ __managed__ cufftCallbackStoreC fft_norm_shift_eq_ptr = fft_norm_shift_eq_cb;
// Shifts
static __device__ __managed__ cufftCallbackStoreC fft_shift_eq_ptr = fft_shift_eq_cb;
static __device__ __managed__ cufftCallbackStoreC fft_shift_ptr = fft_shift_cb;
static __device__ __managed__ cufftCallbackLoadC ifft_shift_ptr = ifft_shift_cb;
static __device__ __managed__ cufftCallbackLoadC ifft_shift_eq_ptr = ifft_shift_eq_cb;
}  // namespace s2fftKernels

#endif