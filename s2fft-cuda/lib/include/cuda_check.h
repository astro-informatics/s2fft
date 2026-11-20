#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

// Checks of CUDA and cuFFT calls. Both throw std::runtime_error, which the FFI handlers
// of extensions.cc turn into ffi::Error::Internal for XLA.

#include <stdexcept>
#include <string>

// Macro to check for CUDA errors. Throws so that the FFI handler can return the failure to XLA.
#define checkCudaErrors(call)                                                                           \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            throw std::runtime_error(std::string("CUDA call \"") + #call + "\" failed at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err));  \
        }                                                                                               \
    } while (0)

// cufft API error checking. Throws so that the FFI handler can return the failure to XLA.
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                                 \
    do {                                                                                                 \
        auto status = static_cast<cufftResult>(call);                                                    \
        if (status != CUFFT_SUCCESS) {                                                                   \
            throw std::runtime_error(std::string("cuFFT call \"") + #call + "\" failed at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " with code " +                    \
                                     std::to_string(static_cast<int>(status)));                          \
        }                                                                                                \
    } while (0)
#endif  // CUFFT_CALL

#endif  // CUDA_CHECK_H
