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

// Macro to check for CUDA errors
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#endif // HRESULT_H
