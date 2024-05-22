#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>

namespace s2fftKernels {

__device__ void computeNphi(int nside, int ring_index, int L, int& nphi, int& offset_ring) {
    // Compute number of pixels
    int total_pixels = 12 * nside * nside;
    int total_rings = 4 * nside - 1;
    int upper_pixels = nside * (nside - 1) * 2;

    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        // Compute lower pixel offset
        nphi = 4 * (total_rings - ring_index);
        nphi = nphi == 0 ? 4 : nphi;
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
    }
}

template <typename complex>
__global__ void spectral_folding(complex* data, complex* output, int nside, int L) {
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (current_indx >= (4 * nside - 1)) {
        return;
    }

    int ring_index = current_indx;
    // Compute nphi of current ring
    int nphi(0);
    int ring_offset(0);
    computeNphi(nside, ring_index, L, nphi, ring_offset);

    // ring index

    int ftm_offset = ring_index * (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring

    int slice_start = (L - nphi / 2);
    int slice_end = slice_start + nphi;

    // Fill up the healpix ring
    for (int i = 0; i < nphi; i++) {
        int folded_index = i + ring_offset;
        int target_index = i + ftm_offset + slice_start;

        output[folded_index] = data[target_index];
    }
    // fold the negative part of the spectrum
    for (int i = 0; i < slice_start; i++) {
        int folded_index = -(1 + i) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_start - (1 + i);

        folded_index = folded_index + ring_offset;
        target_index = target_index + ftm_offset;
        output[folded_index].x += data[target_index].x;
        output[folded_index].y += data[target_index].y;
    }
    // fold the positive part of the spectrum
    for (int i = 0; i < L - nphi / 2; i++) {
        int folded_index = i % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_end + i;

        folded_index = folded_index + ring_offset;
        target_index = target_index + ftm_offset;
        output[folded_index].x += data[target_index].x;
        output[folded_index].y += data[target_index].y;
    }
}
template <typename complex>
__global__ void spectral_folding_parallel(complex* data, complex* output, int nside, int L) {
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute nphi of current ring
    int nphi(0);
    int offset_ring(0);
    computeNphi(nside, current_indx, L, nphi, offset_ring);

    // ring index
    int ring_index = current_indx / (2 * L);
    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    int ftm_offset = ring_index * (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring

    int slice_start = (L - nphi / 2);
    int slice_end = slice_start + nphi;

    // Fill up the healpix ring
    if (offset >= slice_start && offset < slice_end) {
        int center_offset = offset - slice_start;
        int indx = center_offset + offset_ring;

        output[indx] = data[current_indx];
    }
    __syncthreads();
    // fold the negative part of the spectrum
    if (offset < slice_start && true) {
        int folded_index = -(1 + offset) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_start - (1 + offset);

        folded_index = folded_index + offset_ring;
        target_index = target_index + ftm_offset;
        atomicAdd(&output[folded_index].x, data[target_index].x);
        atomicAdd(&output[folded_index].y, data[target_index].y);
    }
    // fold the positive part of the spectrum
    __syncthreads();
    if (offset >= slice_end && true) {
        int folded_index = (offset - slice_end) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_end + (offset - slice_end);

        folded_index = folded_index + offset_ring;
        target_index = target_index + ftm_offset;
        atomicAdd(&output[folded_index].x, data[target_index].x);
        atomicAdd(&output[folded_index].y, data[target_index].y);
    }
}

template <typename complex>
__global__ void spectral_extension(complex* data, complex* output, int nside, int L) {
    // few inits
    int ftm_size = 2 * L;
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;

    if (current_indx >= (4 * nside - 1) * ftm_size) {
        return;
    }
    // Compute nphi of current ring
    int nphi(0);
    int offset_ring(0);
    // ring index
    int ring_index = current_indx / (2 * L);
    computeNphi(nside, ring_index, L, nphi, offset_ring);

    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring

    if (offset < L - nphi / 2) {
        int indx = (-(L - nphi / 2 - offset)) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
    }

    // Compute the central part of the spectrum
    else if (offset >= L - nphi / 2 && offset < L + nphi / 2) {
        int center_offset = offset - /*negative part offset*/ (L - nphi / 2);
        int indx = center_offset + offset_ring;
        output[current_indx] = data[indx];
    }
    // Compute the positive part of the spectrum
    else {
        int reverse_offset = ftm_size - offset;
        int indx = (L - (int)((nphi + 1) / 2) - reverse_offset) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
    }
}

template <typename complex>
HRESULT launch_spectral_folding(complex* data, complex* output, const int& nside, const int& L,
                                cudaStream_t stream) {
    int block_size = 128;
    int ftm_elements = (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    spectral_folding<complex><<<grid_size, block_size, 0, stream>>>(data, output, nside, L);
    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

template <typename complex>
HRESULT launch_spectral_extension(complex* data, complex* output, const int& nside, const int& L,
                                  cudaStream_t stream) {
    // Launch the kernel
    int block_size = 128;
    int ftm_elements = 2 * L * (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    spectral_extension<complex><<<grid_size, block_size, 0, stream>>>(data, output, nside, L);

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

// Specializations
template HRESULT launch_spectral_folding<cufftComplex>(cufftComplex* data, cufftComplex* output,
                                                       const int& nside, const int& L, cudaStream_t stream);
template HRESULT launch_spectral_folding<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                             cufftDoubleComplex* output, const int& nside,
                                                             const int& L, cudaStream_t stream);

template HRESULT launch_spectral_extension<cufftComplex>(cufftComplex* data, cufftComplex* output,
                                                         const int& nside, const int& L, cudaStream_t stream);
template HRESULT launch_spectral_extension<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                               cufftDoubleComplex* output, const int& nside,
                                                               const int& L, cudaStream_t stream);

}  // namespace s2fftKernels