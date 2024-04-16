#ifndef _S2FFT_CALLBACKS_CUH_
#define _S2FFT_CALLBACKS_CUH_

#include <cufft.h>
#include <cstddef>
#include <cuda_runtime.h>

#define FFT_NORM(shift) (shift ? s2fftKernels::fft_norm_shift_ptr : s2fftKernels::fft_norm_noshift_ptr)
#define FFT_NORM_ORTHO(shift) \
    (shift ? s2fftKernels::fft_norm_ortho_shift_ptr : s2fftKernels::fft_norm_ortho_noshift_ptr)

#define FFT_NORM_EQ(shift) (shift ? s2fftKernels::fft_norm_shift_eq_ptr : s2fftKernels::fft_norm_noshift_ptr)
#define FFT_NORM_ORTHO_EQ(shift) \
    (shift ? s2fftKernels::fft_norm_ortho_shift_eq_ptr : s2fftKernels::fft_norm_ortho_noshift_ptr)

namespace s2fftKernels {

static __device__ void fft_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                    void *sharedPointer) {
    int *params = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;
    int n = params[0];
    int dist = params[1];
    int first_element_offset = offset < dist ? 0 : dist;

    float half_f = n / 2.0f;
    int half = ceil(half_f);
    int indx = ((offset + half) % n) + first_element_offset;
    data[indx] = element;

}

static __device__ void fft_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                       void *sharedPointer) {
    int *params = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;
    int n = params[0];
    int first_element_offset = (offset / n) * n;

    float half_f = n / 2.0f;
    int half = ceil(half_f);
    int indx = ((offset + half) % n) + first_element_offset;
    data[indx] = element;
}

static __device__ void fft_norm_ortho_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                         void *sharedPointer) {
    int *size = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;
    int n = *size;
    float norm_factor = 1.0f / sqrtf((float)n);
    element.x /= norm_factor;
    element.y /= norm_factor;
    data[offset] = element;
}

static __device__ void fft_norm_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                   void *sharedPointer) {
    int *size = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataOut;
    int n = *size;
    float norm_factor = 1.0f / (float)n;
    element.x *= norm_factor;
    element.y *= norm_factor;
    data[offset] = element;
}

// Declare the callbacks with shifts

static __device__ void fft_norm_ortho_shift_cb(void *dataOut, size_t offset, cufftComplex element,
                                               void *callerInfo, void *sharedPointer) {
    fft_norm_ortho_cb(dataOut, offset, element, callerInfo, sharedPointer);
    fft_shift_cb(dataOut, offset, element, callerInfo, sharedPointer);
}

static __device__ void fft_norm_shift_cb(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                         void *sharedPointer) {
    fft_norm_cb(dataOut, offset, element, callerInfo, sharedPointer);
    fft_shift_cb(dataOut, offset, element, callerInfo, sharedPointer);
}

static __device__ void fft_norm_ortho_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                                  void *callerInfo, void *sharedPointer) {
    fft_norm_ortho_cb(dataOut, offset, element, callerInfo, sharedPointer);
    fft_shift_eq_cb(dataOut, offset, element, callerInfo, sharedPointer);
}

static __device__ void fft_norm_shift_eq_cb(void *dataOut, size_t offset, cufftComplex element,
                                            void *callerInfo, void *sharedPointer) {
    fft_norm_cb(dataOut, offset, element, callerInfo, sharedPointer);
    fft_shift_eq_cb(dataOut, offset, element, callerInfo, sharedPointer);
}

// Inverse shifts

// Load callback because it has to be done before the inverse fft
static __device__ cufftComplex ifft_shift_cb(void *dataIn, size_t offset, void *callerInfo,
                                             void *sharedPointer) {
    int *params = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataIn;
    int n = params[0];
    int dist = params[1];
    int first_element_offset = offset < dist ? 0 : dist;

    float half_f = n / 2.0f;
    int half = floor(half_f);
    int indx = ((offset + half) % n) + first_element_offset;

    return data[indx];
}

static __device__ cufftComplex ifft_shift_eq_cb(void *dataIn, size_t offset, void *callerInfo,
                                                void *sharedPointer) {
    int *params = (int *)callerInfo;
    cufftComplex *data = (cufftComplex *)dataIn;
    int n = params[0];
    int first_element_offset = (offset / n) * n;

    float half_f = n / 2.0f;
    int half = ceil(half_f);
    int indx = ((offset + half) % n) + first_element_offset;

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