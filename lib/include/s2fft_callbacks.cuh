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

static __device__ int64 fft_shift(size_t offset, int64* params) {
    int64 n = params[0];
    int64 dist = params[1];
    int64 first_element_offset = offset < dist ? 0 : dist;

    int64 half = n / 2;
    //int64 half = ceil(half_f);
    int64 normalized_offset = offset - first_element_offset;
    int64 indx = ((normalized_offset + half) % n) + first_element_offset;

    return indx;
}

static __device__ int64 fft_shift_eq(size_t offset, int64* params) {
    int64 n = params[0];
    int64 first_element_offset = (offset / n) * n;
    int64 offset_in_ring = first_element_offset + offset % n;

    int64 half = n / 2;
    //int64 half = ceil(half_f);
    size_t indx = ((offset_in_ring + half) % n) + first_element_offset;

    return indx;
}

static __device__ void normalize(cufftComplex* element, int64 size) {
    float norm_factor = 1.0f / (float)size;
    element->x *= norm_factor;
    element->y *= norm_factor;
}

static __device__ void normalize_ortho(cufftComplex* element, int64 size) {
    float norm_factor = 1.0f / sqrtf((float)size);
    element->x *= norm_factor;
    element->y *= norm_factor;
}

// Callbacks

static __device__ void fft_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                    void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    int64 indx = fft_shift(offset, params);
    data[indx] = element;
}

static __device__ void fft_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                       void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    int64 indx = fft_shift_eq(offset, params);
    data[indx] = element;
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
    int64 indx = fft_shift(offset, params);

    data[indx] = element;
}

static __device__ void fft_norm_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                         void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize(&element, params[0]);
    int64 indx = fft_shift(offset, params);

    data[indx] = element;
}

static __device__ void fft_norm_ortho_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                                  void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;    
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize_ortho(&element, params[0]);
    int64 indx = fft_shift_eq(offset, params);

    data[indx] = element;
}

static __device__ void fft_norm_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                            void *callerInfo, void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;

    normalize(&element, params[0]);
    int64 indx = fft_shift_eq(offset, params);

    data[indx] = element;
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

    int64 half = n / 2;
    size_t indx = ((normalized_offset + half) % n) + first_element_offset;

    return data[indx];
}

static __device__ cufftComplex ifft_shift_eq_cb(void *dataIn, size_t offset, void *callerInfo,
                                                void *sharedPointer) {
    int64 *params = (int64 *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataIn;
    int64 n = params[0];
    int64 first_element_offset = (offset / n) * n;
    int64 offset_in_ring = first_element_offset + offset % n;

    // printf("offset: %lld, equator_offset: %lld, first_element_offset: %lld\n", offset, equator_offset,
    //        first_element_offset);

    int64 half = n / 2;
    size_t indx = ((offset_in_ring + half) % n) + first_element_offset;

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