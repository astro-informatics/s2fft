#ifndef HRESULT_H
#define HRESULT_H

typedef long HRESULT;

#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)

#define S_OK ((HRESULT)0x00000000L)
#define S_FALSE ((HRESULT)1L)

#define E_ABORT ((HRESULT)0x80004004L)
#define E_ACCESSDENIED ((HRESULT)0x80070005L)
#define E_FAIL ((HRESULT)0x80004005L)
#define E_HANDLE ((HRESULT)0x80070006L)
#define E_INVALIDARG ((HRESULT)0x80070057L)
#define E_NOINTERFACE ((HRESULT)0x80004002L)
#define E_NOTIMPL ((HRESULT)0x80004001L)
#define E_OUTOFMEMORY ((HRESULT)0x8007000EL)
#define E_POINTER ((HRESULT)0x80004003L)
#define E_UNEXPECTED ((HRESULT)0x8000FFFFL)
#define E_OUTOFMEMORY ((HRESULT)0x8007000EL)
#define E_NOTIMPL ((HRESULT)0x80004001L)

#include <stdexcept>
#include <string>

// Macro to check for CUDA errors. Throws so that the FFI handler can return the failure to XLA.
#define checkCudaErrors(call)                                                                       \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            throw std::runtime_error(std::string("CUDA call \"") + #call + "\" failed at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
        }                                                                                           \
    } while (0)

// cufft API error checking. Throws so that the FFI handler can return the failure to XLA.
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                              \
    do {                                                                                              \
        auto status = static_cast<cufftResult>(call);                                                 \
        if (status != CUFFT_SUCCESS) {                                                                \
            throw std::runtime_error(std::string("cuFFT call \"") + #call + "\" failed at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " with code " +                 \
                                     std::to_string(static_cast<int>(status)));                       \
        }                                                                                             \
    } while (0)
#endif  // CUFFT_CALL

#endif  // HRESULT_H
