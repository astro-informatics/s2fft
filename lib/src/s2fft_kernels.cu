#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>

namespace s2fftKernels {

__global__ void spectral_extension(cufftComplex* data, cufftComplex* output, int nside, int L,
                                   int equatorial_ring_num) {
    // Copy data into registers
    int _nside = nside;
    int _L = L;
    int _equatorial_ring_num = equatorial_ring_num;
    // Kernel is launched with a block size of flm_size
    // the ftm size is 2 * L
    // In this case the block size is 2 * L and the blockId is the ring number

    // Compute nphi of current ring
    int nphi(0);

    // Upper Polar rings
    if (blockIdx.x < _nside - 1) {
        nphi = 4 * (blockIdx.x + 1);

    }
    // Lower Polar rings
    else if (blockIdx.x < 2 * _nside + _equatorial_ring_num - 1) {
        nphi = 4 * (_nside - (blockIdx.x - _nside));
    }
    // Equatorial ring
    else {
        nphi = 4 * _nside;
    }

    extern __shared__ cufftComplex ring[];

    // Load the ring into shared memory
    if (threadIdx.x > (_L - nphi) / 2 && threadIdx.x < (_L + nphi) / 2) {
        // Data is guaranteed to be accessed without data races (no need for atomic operations)
        ring[threadIdx.x - (_L - nphi) / 2] = data[blockIdx.x * _L + threadIdx.x];
    }

    __syncthreads();

    // Spectral extension
    // The resulting array has size 2 * L and it has these indices :

    // fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
    // fm,
    // fm[jnp.arange(L - (nphi + 1) // 2) % nphi],

    // The first part of the array is the negative part of the spectrum
    // The second part is the original spectrum
    // The third part is the positive part of the spectrum

    // Compute the negative part of the spectrum
    if (threadIdx.x < (_L - nphi) / 2) {
        int index = (threadIdx.x + 1) % nphi;
        index = nphi - index;
        output[blockIdx.x * _L + threadIdx.x] = ring[index];
    }
    // Compute the central part of the spectrum
    else if (threadIdx.x >= (_L - nphi) / 2 && threadIdx.x < (_L + nphi) / 2) {
        output[blockIdx.x * _L + threadIdx.x] = ring[threadIdx.x - (_L - nphi) / 2];
    }
    // Compute the positive part of the spectrum
    else {
        int index = (threadIdx.x - (_L + nphi) / 2) % nphi;
        output[blockIdx.x * _L + threadIdx.x] = ring[index];
    }
}

HRESULT launch_spectral_extension(cufftComplex* data, cufftComplex* output, const int& nside, const int& L,
                                  const int& equatorial_ring_num, cudaStream_t stream) {
    // Launch the kernel

    int block_size = 2 * L;
    int grid_size = 2 * nside + equatorial_ring_num;

    spectral_extension<<<grid_size, block_size, block_size * sizeof(cufftComplex), stream>>>(
            data, output, nside, L, equatorial_ring_num);

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

}  // namespace s2fftKernels