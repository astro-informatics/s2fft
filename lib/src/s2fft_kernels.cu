#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>

namespace s2fftKernels {

__global__ void spectral_folding(cufftComplex* data, cufftComplex* output, int nside, int L,
                                 int equatorial_offset_start, int equatorial_offset_end) {}

__global__ void spectral_extension(cufftComplex* data, cufftComplex* output, int nside, int L,
                                   int equatorial_offset_start, int equatorial_offset_end) {
    // few inits
    int polar_rings(nside - 1);
    int equator_rings(3 * nside + 1);
    int total_rings(4 * nside - 1);
    int total_pixels(12 * nside * nside);
    // Which ring are we working on
    int current_indx(blockIdx.x * blockDim.x + threadIdx.x);
    int ring_index(0);
    int offset(0);
    int pos(0);

    // Upper Polar rings
    // if (current_indx < equatorial_offset_start) {
    //    ring_index = current_indx / 4;
    //    pos = 1;
    //    // Lower Polar rings
    //} else if (current_indx >= equatorial_offset_end) {
    //    // Compute ring_index from the end
    //    ring_index = (total_pixels - current_indx) / 4;
    //    // Compute ring_index from the start
    //    ring_index = total_rings - ring_index;
    //    pos = -1;
    //    // Equatorial ring
    //} else {
    //    int offset_in_equator_matrix = current_indx - equatorial_offset_start;
    //    // Ring index in the equator matrix
    //    ring_index = offset_in_equator_matrix / (4 * nside);
    //    // Ring index in the total Healspix array
    //    ring_index = ring_index + polar_rings;
    //    pos = 0;
    //}
    ring_index = current_indx / (2 * L);
    offset = current_indx % (2 * L);

    // Compute nphi of current ring
    int nphi(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        pos = 1;

    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index);
        pos = -1;
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        pos = 0;
    }

    // Spectral extension
    // The resulting array has size 2 * L and it has these indices :

    // fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
    // fm,
    // fm[jnp.arange(L - (nphi + 1) // 2) % nphi],

    // Compute the negative part of the spectrum
    if (offset < L - nphi / 2) {
        int indx = (-(L - nphi / 2 + offset)) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        output[current_indx] = data[indx];
        printf("Negative part: element at offset %d  came from %d\n", current_indx, indx);
    }
    // Compute the central part of the spectrum
    else if (offset >= L - nphi / 2 && offset < L + nphi / 2) {
        int center_offset = (L - nphi / 2);
        int indx = current_indx - (L - nphi) / 2;
        output[current_indx] = data[indx];
        printf("Central part: element at offset %d came from %d\n", current_indx, indx);
    }
    // Compute the positive part of the spectrum
    else {
        int indx = (offset - (L + nphi) / 2) % nphi;
        output[current_indx] = data[indx];
        printf("Positive part: element at offset %d came from %d\n", current_indx, indx);
    }

    // Only use global memory for now
    printf("For current index %d, ring index is %d and nphi is %d and pos is %d\n", current_indx, ring_index,
           nphi, pos);
}

HRESULT launch_spectral_folding(cufftComplex* data, cufftComplex* output, const int& nside, const int& L,
                                const int64& equatorial_offset_start, const int64& equatorial_offset_end,
                                cudaStream_t stream) {}

HRESULT launch_spectral_extension(cufftComplex* data, cufftComplex* output, const int& nside, const int& L,
                                  const int64& equatorial_offset_start, const int64& equatorial_offset_end,
                                  cudaStream_t stream) {
    // Launch the kernel
    std::cout << "Launching kernel" << std::endl;
    int block_size = 128;
    int ftm_elements = 2 * L * (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;
    std::cout << "Grid size: " << grid_size << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "equatorial_offset_start: " << equatorial_offset_start << std::endl;
    std::cout << "equatorial_offset_end: " << equatorial_offset_end << std::endl;

    spectral_extension<<<grid_size, block_size, 0, stream>>>(data, output, nside, L, equatorial_offset_start,
                                                             equatorial_offset_end);

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

}  // namespace s2fftKernels