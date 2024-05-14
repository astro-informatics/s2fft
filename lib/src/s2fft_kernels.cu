#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>



namespace s2fftKernels {

//__device__ void spectral_extension_with_shared_mem(cufftComplex* data, cufftComplex* output, int nphi,
//                                                   int _nside, int _L, int equatorial_ring_num)
//
//{
//    extern __shared__ cufftComplex ring[];
//
//    // Load the ring into shared memory
//    if (threadIdx.x > (_L - nphi) / 2 && threadIdx.x < (_L + nphi) / 2) {
//        // Data is guaranteed to be accessed without data races (no need for atomic operations)
//        ring[threadIdx.x - (_L - nphi) / 2] = data[blockIdx.x * _L + threadIdx.x];
//    }
//
//    __syncthreads();
//
//    // Spectral extension
//    // The resulting array has size 2 * L and it has these indices :
//
//    // fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
//    // fm,
//    // fm[jnp.arange(L - (nphi + 1) // 2) % nphi],
//
//    // The first part of the array is the negative part of the spectrum
//    // The second part is the original spectrum
//    // The third part is the positive part of the spectrum
//
//    // Compute the negative part of the spectrum
//    if (threadIdx.x < (_L - nphi) / 2) {
//        int index = (threadIdx.x + 1) % nphi;
//        index = nphi - index;
//        output[blockIdx.x * _L + threadIdx.x] = ring[index];
//    }
//    // Compute the central part of the spectrum
//    else if (threadIdx.x >= (_L - nphi) / 2 && threadIdx.x < (_L + nphi) / 2) {
//        output[blockIdx.x * _L + threadIdx.x] = ring[threadIdx.x - (_L - nphi) / 2];
//    }
//    // Compute the positive part of the spectrum
//    else {
//        int index = (threadIdx.x - (_L + nphi) / 2) % nphi;
//        output[blockIdx.x * _L + threadIdx.x] = ring[index];
//    }
//}

//__device__ void spectral_extension_from_global_mem(cufftComplex* data, cufftComplex* output, int nphi,
//                                                   int _nside, int L, int equatorial_offset_num) {
//    // Spectral extension
//    // The resulting array has size 2 * L and it has these indices :
//
//    // fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
//    // fm,
//    // fm[jnp.arange(L - (nphi + 1) // 2) % nphi],
//
//    // The first part of the array is the negative part of the spectrum
//    // The second part is the original spectrum
//    // The third part is the positive part of the spectrum
//
//    // Compute the negative part of the spectrum
//    // L is equal to blockDim.x
//    int current_output_index = blockIdx.x * blockDim.x + threadIdx.x;
//    int first_element_in_ring_offset(0);
//    int ftm_size = 2 * L;
//    int ring_index(0);
//
//    // Compute the ring index
//    // Upper Polar rings
//    if (blockIdx.x < _nside - 1) {
//        ring_index = nphi / 4;
//        // Lower Polar rings
//    } else if (blockIdx.x > 3 * _nside - 1) {
//        // Equatorial ring
//    } else {
//        int offset_in_equator_matrx = ring_index = 4 * _nside - 2 - blockIdx.x;
//    }
//
//    if (threadIdx.x < L - nphi / 2) {
//        // L - nphi // 2 + offset
//        int negative_part_offset = 0;
//        int indx = (-(L - nphi / 2 + threadIdx.x)) % nphi;
//        // Make sure that python % and C % are the same
//        // https://stackoverflow.com/questions/3883004/how-does-the-modulo-operator-work-on-negative-numbers-in-python
//        indx = indx < 0 ? nphi + indx : indx;
//        // output[current_index] = data[indx];
//        printf("Negative part: thread %d element at offset %d negative part offset %d came from %d\n",
//               threadIdx.x, current_output_index, negative_part_offset, indx);
//    }
//    // Compute the central part of the spectrum
//    else if (threadIdx.x >= L - nphi / 2 && threadIdx.x < L + nphi / 2) {
//        // -6
//        int center_offset = (L - nphi / 2);
//        int indx =
//
//                printf("Central part: %d element at offset %d came from %d\n", threadIdx.x,
//                       (blockIdx.x * L + threadIdx.x), (blockIdx.x * L + threadIdx.x - (_L - nphi) / 2));
//    }
//    // Compute the positive part of the spectrum
//    else {
//        int index = (threadIdx.x - (L + nphi) / 2) % nphi;
//        output[blockIdx.x * L + threadIdx.x] = data[blockIdx.x * _L + index];
//        printf("Positive part: %d element at offset %d came from %d\n", threadIdx.x,
//               (blockIdx.x * _L + threadIdx.x), (blockIdx.x * _L + index));
//    }
//}

__global__ void spectral_extension(cufftComplex* data, cufftComplex* output, int nside, int L,
                                   int equatorial_offset_start, int equatorial_offset_end) {
    
    // few inits
    int polar_rings(nside -1);
    int equator_rings(3 * nside + 1);
    int total_rings(4 * nside - 1);
    int total_pixels(12 * nside * nside);
    // Which ring are we working on
    int current_indx(blockIdx.x * blockDim.x + threadIdx.x);
    int ring_index(0);
    int pos(0);

    // Upper Polar rings    
    if (current_indx < equatorial_offset_start) {
        ring_index = current_indx / 4;
        pos = 1;
    // Lower Polar rings
    } else if (current_indx >= equatorial_offset_end) {
        ring_index = (total_pixels - current_indx) / 4;
        pos = -1;
    // Equatorial ring
    } else {
        int offset_in_equator_matrix = current_indx - equatorial_offset_start;
        // Ring index in the equator matrix
        ring_index = offset_in_equator_matrix / (4 * nside);
        // Ring index in the total Healspix array
        ring_index = ring_index + polar_rings;
        pos = 0;

    }

    // Compute nphi of current ring
    int nphi(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (blockIdx.x + 1);

    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index + 1);
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
    }

    // When nphi is equal or more than L, shared memory is not very useful
    // In this case we can use global memory
    printf("For current index %d, ring index is %d and nphi is %d and pos is %d\n", current_indx, ring_index, nphi, pos);
}

HRESULT launch_spectral_extension(cufftComplex* data, cufftComplex* output, const int& nside, const int& L,
                                  const int64& equatorial_offset_start,  const int64& equatorial_offset_end, cudaStream_t stream) {
    // Launch the kernel
    std::cout << "Launching kernel" << std::endl;
    int block_size = 2 * L;
    int grid_size = 4 * nside - 1;
    std::cout << "Grid size: " << grid_size << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "equatorial_offset_start: " << equatorial_offset_start << std::endl;
    std::cout << "equatorial_offset_end: " << equatorial_offset_end << std::endl;

    spectral_extension<<<grid_size, block_size, block_size * sizeof(cufftComplex), stream>>>(
            data, output, nside, L, equatorial_offset_start,equatorial_offset_end);

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

}  // namespace s2fftKernels