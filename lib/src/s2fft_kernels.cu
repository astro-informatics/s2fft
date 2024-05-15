#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>

namespace s2fftKernels {

__global__ void spectral_folding(int* data, int* output, int nside, int L, int equatorial_offset_start,
                                 int equatorial_offset_end) {
    // few inits
    int polar_rings = nside - 1;
    int equator_rings = 3 * nside + 1;
    int total_rings = 4 * nside - 1;
    int ftm_size = 2 * L;
    // Compute number of pixels
    int total_pixels = 12 * nside * nside;
    int upper_pixels = nside * (nside - 1) * 2;
    int equator_pixels = 4 * nside * equator_rings;
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos(0);
    int indx(0);
    // Compute nphi of current ring
    int nphi(0);

    // ring index
    int ring_index = current_indx / (2 * L);
    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring
    int offset_ring(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
        pos = 1;
    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index);
        // Compute lower pixel offset
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
        pos = -1;
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
        pos = 0;
    }

    int slice_start = offset + (L - nphi / 2);
    int slice_end = offset + nphi;
}

__global__ void spectral_extension(int* data, int* output, int nside, int L, int equatorial_offset_start,
                                   int equatorial_offset_end) {
    // few inits
    int polar_rings = nside - 1;
    int equator_rings = 3 * nside + 1;
    int total_rings = 4 * nside - 1;
    int ftm_size = 2 * L;
    // Compute number of pixels
    int total_pixels = 12 * nside * nside;
    int upper_pixels = nside * (nside - 1) * 2;
    int equator_pixels = 4 * nside * equator_rings;
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos(0);
    int indx(0);
    // Compute nphi of current ring
    int nphi(0);

    // ring index
    int ring_index = current_indx / (2 * L);
    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring
    int offset_ring(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
        pos = 1;
    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index);
        // Compute lower pixel offset
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
        pos = -1;
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
        pos = 0;
    }

    // Spectral extension
    // The resulting array has size 2 * L and it has these indices :

    // fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
    // fm,
    // fm[jnp.arange(L - (nphi + 1) // 2) % nphi],

    // Compute the negative part of the spectrum
    // printf("Offset %d (L + nphi / 2) %d \n", offset, (L + nphi / 2));
    int ring_print = 12;
    if (offset < L - nphi / 2) {
        indx = (-(L - nphi / 2 - offset)) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
        if (ring_index == ring_print)
            printf("Negative part: current thread %d, offset %d, ring index %d, ring offset %d, nphi %d , "
                   "target index %d, input element %d output element %d\n",
                   current_indx, offset, ring_index, offset_ring, nphi, indx, data[indx],
                   output[current_indx]);
    }

    // Compute the central part of the spectrum
    else if (offset >= L - nphi / 2 && offset < L + nphi / 2) {
        int center_offset = offset - /*negative part offset*/ (L - nphi / 2);
        indx = center_offset + offset_ring;
        output[current_indx] = data[indx];
        if (ring_index == ring_print)
            printf("Center part: current thread %d, offset %d, center offset %d, ring index %d, ring offset "
                   "%d, nphi %d "
                   "target index %d, input element %d output element %d\n",
                   current_indx, offset, center_offset, ring_index, offset_ring, nphi, indx, data[indx],
                   output[current_indx]);
    }
    // Compute the positive part of the spectrum
    else {
        int reverse_offset = ftm_size - offset;
        indx = (L - (int)((nphi + 1) / 2) - reverse_offset) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
        if (ring_index == ring_print)
            printf("Positive part: current thread %d, offset %d, ring index %d, ring offset "
                   "%d, nphi %d  "
                   "target index %d, input element %d output element %d\n",
                   current_indx, offset, ring_index, offset_ring, nphi, indx, data[indx],
                   output[current_indx]);
        // printf("Positive part: element at offset %d came from %d\n", current_indx, indx);
    }

    // Only use global memory for now
    // printf("For current index %d,data index %d ring index is %d and nphi is %d and pos is %d, output is
    // [%d] "
    //        "and original is [%d]\n",
    //        current_indx, indx, ring_index, nphi, pos, output[current_indx], data[indx]);
    //}
}

HRESULT launch_spectral_folding(int* data, int* output, const int& nside, const int& L,
                                const int64& equatorial_offset_start, const int64& equatorial_offset_end,
                                cudaStream_t stream) {
    return E_NOTIMPL;
}

HRESULT launch_spectral_extension(int* data, int* output, const int& nside, const int& L,
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