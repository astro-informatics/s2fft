#ifndef CUDADEVICEGUARD_HPP
#define CUDADEVICEGUARD_HPP

#include <cuda_runtime.h>

/**
 * @brief RAII selection of the device the current scope operates on.
 *
 * XLA may run FFI handlers from host threads whose current device is not the device owning
 * the buffers (single-process multi-GPU, astro-informatics/s2fft#389). cuFFT plans and
 * streams bind to the device that is current when they are created, so the execution device
 * must be selected before any plan or stream is touched. The previous device is restored on
 * scope exit; a negative ordinal is a no-op (host-side / device-agnostic callers).
 */
class CudaDeviceScope {
public:
    explicit CudaDeviceScope(int device) : m_restore(-1) {
        if (device >= 0) {
            int current = -1;
            if (cudaGetDevice(&current) == cudaSuccess && current != device) {
                cudaSetDevice(device);
                m_restore = current;
            }
        }
    }

    ~CudaDeviceScope() {
        if (m_restore >= 0) {
            cudaSetDevice(m_restore);
        }
    }

    CudaDeviceScope(const CudaDeviceScope &) = delete;
    CudaDeviceScope &operator=(const CudaDeviceScope &) = delete;

private:
    int m_restore;
};

#endif  // CUDADEVICEGUARD_HPP
