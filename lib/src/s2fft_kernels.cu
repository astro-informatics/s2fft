#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>
#include <cooperative_groups.h>

namespace s2fftKernels {

__device__ void compute_nphi_offset_from_ring(int nside, int ring_index, int L, int& nphi, int& offset_ring) {
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

__device__ int ncap(int nside) {
    return 2 * nside * (nside - 1);
}

__device__ int npix(int nside) {
    return 12 * nside * nside;
}

__device__ int rmax(int nside) {
    return 4 * nside - 2;
}

__device__ int compute_nphi_from_ring(int r, int nside) {
    if (r < nside) {
        return 4 * (r + 1);
    } else if (r < 3 * nside) {
        return 4 * nside;
    } else {
        return 4 * (rmax(nside) - r + 1);
    }
}

template <typename T>
__device__ void inline swap(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}
template <typename complex>
__global__ void spectral_folding(complex* data, complex* output, int nside, int L, bool shift) {
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (current_indx >= (4 * nside - 1)) {
        return;
    }

    int ring_index = current_indx;
    // Compute nphi of current ring
    int nphi(0);
    int ring_offset(0);
    compute_nphi_offset_from_ring(nside, ring_index, L, nphi, ring_offset);

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

    if (shift) {
        int half_nphi = nphi / 2;
        // Shift the spectrum
        for (int i = 0; i < half_nphi; i++) {
            int origin_index = i + ring_offset;
            int shifted_index = origin_index + half_nphi;
            swap(output[origin_index], output[shifted_index]);
        }
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
    compute_nphi_offset_from_ring(nside, ring_index, L, nphi, offset_ring);

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
                                const bool& shift, cudaStream_t stream) {
    int block_size = 128;
    int ftm_elements = (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    spectral_folding<complex><<<grid_size, block_size, 0, stream>>>(data, output, nside, L, shift);
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
                                                       const int& nside, const int& L, const bool& shift,
                                                       cudaStream_t stream);
template HRESULT launch_spectral_folding<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                             cufftDoubleComplex* output, const int& nside,
                                                             const int& L, const bool& shift,
                                                             cudaStream_t stream);


template HRESULT launch_spectral_extension<cufftComplex>(cufftComplex* data, cufftComplex* output,
                                                         const int& nside, const int& L, cudaStream_t stream);
template HRESULT launch_spectral_extension<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                               cufftDoubleComplex* output, const int& nside,
                                                               const int& L, cudaStream_t stream);


// New shift/normalize kernel implementation


__device__ void pixel_to_ring_offset_nphi(long long int p, int nside, int& r, int& o , int& nphi) {
    long long int Ncap = ncap(nside);
    long long int Npix = npix(nside);
    int Rmax = rmax(nside);

    if (p < Ncap) { // Upper Polar Cap
        double p_d = static_cast<double>(p);
        int k = static_cast<int>(floor(0.5 * (1.0 + sqrt(1.0 + 2.0 * p_d)))) - 1;
        r = k;
        o = p - 2 * k * (k + 1);
        nphi = 4 * (k + 1);
    } else if (p < Npix - Ncap) { // Equatorial Belt
        long long int q = p - Ncap;
        int k = q / (4 * nside); // Integer division, floor is implicit and correct
        r = (nside - 1) + k;
        o = q % (4 * nside);
        nphi = 4 * nside;
    } else { // Lower Polar Cap
        long long int pprime = Npix - 1 - p;
        double pprime_d = static_cast<double>(pprime);
        int k = static_cast<int>(floor(0.5 * (1.0 + sqrt(1.0 + 2.0 * (pprime_d + 1.0))))) - 1;
        r = (3 * nside - 1) + k; // Ring index from the south pole
        o = 4 * (nside - k - 1) - 1 - (pprime - 2 * k * (k + 1));
        nphi = 4 * (nside - k - 1); // nphi for the south cap
    }
}

__device__ long long int offset_ring_gpu(int r, int nside) {
    long long int Ncap = ncap(nside);
    if (r < nside -1) {
        return 2 * r * (r + 1);
    } else if (r <= 3 * nside - 1) {
        return Ncap + 4 * nside * (r - nside + 1);
    } else {
        long long int Npix = npix(nside);
        int Rmax = rmax(nside);
        int s = Rmax - r;
        return Npix - 2 * s * (s + 1);
    }
}

template <typename complex, typename T>
__global__ void shift_normalize_kernel(complex* data, int nside, bool apply_shift, int norm) {
    long long int p = blockIdx.x * blockDim.x + threadIdx.x;
    long long int Npix = npix(nside);

    if (p >= Npix) return;

    int r, o , nphi;
    pixel_to_ring_offset_nphi(p, nside, r, o, nphi);

    complex element = data[p];

    if (norm == 0) {
        element.x /= nphi;
        element.y /= nphi;
    } else if (norm == 1) {
        T norm_val = sqrt((T)nphi);
        element.x /= norm_val;
        element.y /= norm_val;
    }

    if (apply_shift) {
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        grid.sync();

        long long int ring_start = offset_ring_gpu(r, nside);
        long long int shifted_o = (o + nphi / 2) % nphi;
        long long int dest_p = ring_start + shifted_o;
        data[dest_p] = element;
    } else {
        data[p] = element;
    }
}

template <typename complex>
HRESULT launch_shift_normalize_kernel(
    cudaStream_t stream,
    complex* data,
    int nside,
    bool apply_shift,
    int norm
) {
    long long int Npix = 12 * nside * nside;
    int block_size = 256;
    int grid_size = (Npix + block_size - 1) / block_size;
    std::cout << "Launching shift_normalize_kernel with Npix: " << Npix
              << ", grid_size: " << grid_size << ", block_size: " << block_size << std::endl;

    if constexpr (std::is_same_v<complex, cufftComplex>) {
        shift_normalize_kernel<cufftComplex, float><<<grid_size, block_size, 0, stream>>>((cufftComplex*)data, nside, apply_shift, norm);
    } else {
        shift_normalize_kernel<cufftDoubleComplex, double><<<grid_size, block_size, 0, stream>>>((cufftDoubleComplex*)data, nside, apply_shift, norm);
    }

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

// Specializations
template HRESULT launch_shift_normalize_kernel<cufftComplex>(
    cudaStream_t stream,
    cufftComplex* data,
    int nside,
    bool apply_shift,
    int norm
);

template HRESULT launch_shift_normalize_kernel<cufftDoubleComplex>(
    cudaStream_t stream,
    cufftDoubleComplex* data,
    int nside,
    bool apply_shift,
    int norm
);

}  // namespace s2fftKernels
