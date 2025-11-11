#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <cooperative_groups.h>
#include <iostream>

namespace s2fftKernels {

// ============================================================================
// HELPER DEVICE FUNCTIONS
// ============================================================================

/**
 * @brief Computes the number of pixels in the polar caps for a given Nside.
 *
 * This function calculates the total number of pixels contained within both
 * polar caps (north and south) of a HEALPix sphere for the given Nside parameter.
 *
 * @param nside The HEALPix Nside parameter.
 * @return The number of pixels in both polar caps combined.
 */
__device__ int ncap(int nside) { return 2 * nside * (nside - 1); }

/**
 * @brief Computes the total number of pixels for a given Nside.
 *
 * This function calculates the total number of pixels in a HEALPix sphere
 * for the given Nside parameter.
 *
 * @param nside The HEALPix Nside parameter.
 * @return The total number of pixels (12 * nside^2).
 */
__device__ int npix(int nside) { return 12 * nside * nside; }

/**
 * @brief Computes the maximum ring index for a given Nside.
 *
 * This function calculates the highest ring index in the HEALPix tessellation
 * for the given Nside parameter.
 *
 * @param nside The HEALPix Nside parameter.
 * @return The maximum ring index (4 * nside - 2).
 */
__device__ int rmax(int nside) { return 4 * nside - 2; }

/**
 * @brief Computes the number of pixels and ring offset for a given ring index.
 *
 * This function calculates the number of pixels (nphi) in a specific ring and
 * the offset to the start of that ring in the HEALPix pixel numbering scheme.
 * It handles polar caps and equatorial rings differently according to HEALPix geometry.
 *
 * @param nside The HEALPix Nside parameter.
 * @param ring_index The index of the ring (0-based).
 * @param L The harmonic band limit (unused in current implementation).
 * @param nphi Reference to store the number of pixels in the ring.
 * @param offset_ring Reference to store the offset to the start of the ring.
 */
__device__ void compute_nphi_offset_from_ring(int nside, int ring_index, int L, int& nphi, int& offset_ring) {
    // Step 1: Compute basic HEALPix parameters
    int total_pixels = 12 * nside * nside;
    int total_rings = 4 * nside - 1;
    int upper_pixels = nside * (nside - 1) * 2;

    // Step 2: Determine ring type and compute nphi and offset
    // Use triangular number formula: sum from 0 to n = n * (n + 1) / 2

    // Step 2a: Upper Polar rings (0 to nside-2)
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
    }
    // Step 2b: Lower Polar rings (3*nside to 4*nside-2)
    else if (ring_index > 3 * nside - 1) {
        // Compute lower pixel offset using symmetry
        nphi = 4 * (total_rings - ring_index);
        nphi = nphi == 0 ? 4 : nphi;  // Handle edge case
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
    }
    // Step 2c: Equatorial rings (nside-1 to 3*nside-1)
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
    }
}

/**
 * @brief Converts HEALPix pixel index to ring coordinates and pixel information.
 *
 * This function maps a HEALPix pixel index to its corresponding ring index,
 * offset within the ring, number of pixels in the ring, and the start index
 * of the ring. It correctly handles all three HEALPix regions: upper polar cap,
 * equatorial belt, and lower polar cap.
 *
 * @param p The HEALPix pixel index (0-based).
 * @param nside The HEALPix Nside parameter.
 * @param r Reference to store the ring index.
 * @param o Reference to store the offset within the ring.
 * @param nphi Reference to store the number of pixels in the ring.
 * @param r_start Reference to store the starting pixel index of the ring.
 */
__device__ void pixel_to_ring_offset_nphi(long long int p, int nside, int& r, int& o, int& nphi,
                                          int& r_start) {
    // Step 1: Compute HEALPix parameters
    long long int Ncap = ncap(nside);
    long long int Npix = npix(nside);
    int Rmax = rmax(nside);

    // Step 2: Determine which region the pixel belongs to and compute coordinates
    if (p < Ncap) {
        // Step 2a: Upper Polar Cap
        double p_d = static_cast<double>(p);
        // Use inverse triangular number formula to find ring
        int k = static_cast<int>(floor(0.5 * (sqrt(1.0 + 2.0 * p_d) - 1.0)));
        r = k;
        o = p - 2 * k * (k + 1);
        r_start = 2 * k * (k + 1);
        nphi = 4 * (k + 1);
    } else if (p < Npix - Ncap) {
        // Step 2b: Equatorial Belt
        long long int q = p - Ncap;
        int k = q / (4 * nside);
        r = (nside - 1) + k;
        o = q % (4 * nside);
        o = o < 0 ? 4 * nside + o : o;  // Ensure positive offset
        r_start = Ncap + 4 * nside * k;
        nphi = 4 * nside;
    } else {
        // Step 2c: Lower Polar Cap (use symmetry with upper cap)
        long long int pprime = Npix - 1 - p;
        double pprime_d = static_cast<double>(pprime);
        int k_south = static_cast<int>(floor(0.5 * (sqrt(1.0 + 2.0 * pprime_d) - 1.0)));
        r = Rmax - k_south;
        long long o_prime = pprime - 2 * k_south * (k_south + 1);
        int nphi_lo = 4 * (k_south + 1);
        o = nphi_lo - 1 - o_prime;
        r_start = Npix - (2 * k_south * (k_south + 1) + nphi_lo);
        nphi = nphi_lo;
    }
}

/**
 * @brief Generic inline swap function for device code.
 *
 * This function swaps the values of two variables of any type T.
 * It's used within CUDA kernels for efficient data manipulation.
 *
 * @tparam T The type of the variables to swap.
 * @param a Reference to the first variable.
 * @param b Reference to the second variable.
 */
template <typename T>
__device__ void inline swap(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}

// ============================================================================
// GLOBAL KERNELS
// ============================================================================

/**
 * @brief CUDA kernel for spectral folding in spherical harmonic transforms.
 *
 * This kernel performs spectral folding operations on ring-ordered data,
 * transforming from Fourier coefficient space to HEALPix pixel space.
 * It handles both positive and negative frequency components and applies
 * optional FFT shifting.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input data array containing Fourier coefficients per ring.
 * @param output Output array for folded HEALPix pixel data.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param shift Flag indicating whether to apply FFT shifting.
 */
template <typename complex>
__global__ void spectral_folding(complex* data, complex* output, int nside, int L, bool shift) {
    // Step 1: Determine which ring this thread is processing
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (current_indx >= (4 * nside - 1)) {
        return;
    }

    // Step 2: Initialize ring parameters
    int ring_index = current_indx;
    int nphi(0);
    int ring_offset(0);
    compute_nphi_offset_from_ring(nside, ring_index, L, nphi, ring_offset);

    // Step 3: Compute indices for Fourier coefficient and HEALPix data
    int ftm_offset = ring_index * (2 * L);  // Offset for this ring's FTM data
    int slice_start = (L - nphi / 2);       // Start of central slice
    int slice_end = slice_start + nphi;     // End of central slice

    // Step 4: Copy the central part of the spectrum directly
    for (int i = 0; i < nphi; i++) {
        int folded_index = i + ring_offset;
        int target_index = i + ftm_offset + slice_start;
        output[folded_index] = data[target_index];
    }

    // Step 5: Fold the negative part of the spectrum
    for (int i = 0; i < slice_start; i++) {
        int folded_index = -(1 + i) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_start - (1 + i);

        folded_index = folded_index + ring_offset;
        target_index = target_index + ftm_offset;
        output[folded_index].x += data[target_index].x;
        output[folded_index].y += data[target_index].y;
    }

    // Step 6: Fold the positive part of the spectrum
    for (int i = 0; i < L - nphi / 2; i++) {
        int folded_index = i % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_end + i;

        folded_index = folded_index + ring_offset;
        target_index = target_index + ftm_offset;
        output[folded_index].x += data[target_index].x;
        output[folded_index].y += data[target_index].y;
    }

    // Step 7: Apply FFT shifting if requested
    if (shift) {
        int half_nphi = nphi / 2;
        for (int i = 0; i < half_nphi; i++) {
            int origin_index = i + ring_offset;
            int shifted_index = origin_index + half_nphi;
            swap(output[origin_index], output[shifted_index]);
        }
    }
}

/**
 * @brief CUDA kernel for spectral extension in spherical harmonic transforms.
 *
 * This kernel performs the inverse operation of spectral folding, extending
 * HEALPix pixel data back to full Fourier coefficient space. It maps folded
 * frequency components back to their appropriate positions in the extended spectrum.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input array containing folded HEALPix pixel data.
 * @param output Output array for extended Fourier coefficients per ring.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 */
template <typename complex>
__global__ void spectral_extension(complex* data, complex* output, int nside, int L) {
    // Step 1: Initialize basic parameters
    int ftm_size = 2 * L;
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;

    if (current_indx >= (4 * nside - 1) * ftm_size) {
        return;
    }

    // Step 2: Determine ring and frequency offset
    int ring_index = current_indx / (2 * L);
    int offset = current_indx % (2 * L);  // Frequency offset within this ring

    // Step 3: Get ring parameters
    int nphi(0);
    int offset_ring(0);
    compute_nphi_offset_from_ring(nside, ring_index, L, nphi, offset_ring);

    // Step 4: Map frequency components based on their position in spectrum
    if (offset < L - nphi / 2) {
        // Step 4a: Negative frequency part
        int indx = (-(L - nphi / 2 - offset)) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
    } else if (offset >= L - nphi / 2 && offset < L + nphi / 2) {
        // Step 4b: Central part of the spectrum (direct mapping)
        int center_offset = offset - (L - nphi / 2);
        int indx = center_offset + offset_ring;
        output[current_indx] = data[indx];
    } else {
        // Step 4c: Positive frequency part
        int reverse_offset = ftm_size - offset;
        int indx = (L - (int)((nphi + 1) / 2) - reverse_offset) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output[current_indx] = data[indx];
    }
}

/**
 * @brief CUDA kernel for FFT shifting and normalization of HEALPix data.
 *
 * This kernel applies per-ring normalization and optional FFT shifting to HEALPix
 * pixel data. It processes each pixel independently, computing its ring coordinates
 * and applying the appropriate transformations. Supports both in-place (with cooperative
 * synchronization) and out-of-place (with separate buffer) modes.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @tparam T The floating-point type (float or double) for normalization.
 * @param data Input/output array of HEALPix pixel data.
 * @param shift_buffer Scratch buffer for out-of-place shifting (can be nullptr).
 * @param nside The HEALPix Nside parameter.
 * @param apply_shift Flag indicating whether to apply FFT shifting.
 * @param norm Normalization type (0=by nphi, 1=by sqrt(nphi), 2=no normalization).
 * @param use_out_of_place If true, write shifted data to shift_buffer; if false, use in-place with
 * grid.sync().
 */
template <typename complex, typename T>
__global__ void shift_normalize_kernel(complex* data, complex* shift_buffer, int nside, bool apply_shift,
                                       int norm, bool use_out_of_place) {
    // Step 1: Get pixel index and check bounds
    long long int p = blockIdx.x * blockDim.x + threadIdx.x;
    long long int Npix = npix(nside);

    if (p >= Npix) return;

    // Step 2: Convert pixel index to ring coordinates
    int r = 0, o = 0, nphi = 1, r_start = 0;
    pixel_to_ring_offset_nphi(p, nside, r, o, nphi, r_start);

    // Step 3: Read and normalize the pixel data
    complex element = data[p];

    if (norm == 0) {
        // Step 3a: Normalize by nphi
        element.x /= nphi;
        element.y /= nphi;
    } else if (norm == 1) {
        // Step 3b: Normalize by sqrt(nphi)
        T norm_val = sqrt((T)nphi);
        element.x /= norm_val;
        element.y /= norm_val;
    }
    // Step 3c: No normalization for norm == 2

    // Step 4: Apply FFT shifting if requested
    if (apply_shift) {
        // Step 4a: Compute shifted position within ring
        long long int shifted_o = (o + nphi / 2) % nphi;
        shifted_o = shifted_o < 0 ? nphi + shifted_o : shifted_o;
        long long int dest_p = r_start + shifted_o;

        if (use_out_of_place) {
            // Step 4b: Out-of-place mode - write to separate buffer (no sync needed)
            shift_buffer[dest_p] = element;
        } else {
            // Step 4c: In-place mode - sync then write
            cooperative_groups::grid_group grid = cooperative_groups::this_grid();
            grid.sync();
            data[dest_p] = element;
        }
    } else {
        // Step 4d: No shift - write back to original position
        data[p] = element;
    }
}

// ============================================================================
// C++ LAUNCH FUNCTIONS
// ============================================================================

/**
 * @brief Launches the spectral folding CUDA kernel.
 *
 * This function configures and launches the spectral_folding kernel with
 * appropriate grid and block dimensions. It performs error checking and
 * returns the execution status.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input data array containing Fourier coefficients per ring.
 * @param output Output array for folded HEALPix pixel data.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param shift Flag indicating whether to apply FFT shifting.
 * @param stream CUDA stream for kernel execution.
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_spectral_folding(complex* data, complex* output, const int& nside, const int& L,
                                const bool& shift, cudaStream_t stream) {
    // Step 1: Configure kernel launch parameters
    int block_size = 128;
    int ftm_elements = (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    // Step 2: Launch the kernel
    spectral_folding<complex><<<grid_size, block_size, 0, stream>>>(data, output, nside, L, shift);

    // Step 3: Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

/**
 * @brief Launches the spectral extension CUDA kernel.
 *
 * This function configures and launches the spectral_extension kernel with
 * appropriate grid and block dimensions. It performs error checking and
 * returns the execution status.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param data Input array containing folded HEALPix pixel data.
 * @param output Output array for extended Fourier coefficients per ring.
 * @param nside The HEALPix Nside parameter.
 * @param L The harmonic band limit.
 * @param stream CUDA stream for kernel execution.
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_spectral_extension(complex* data, complex* output, const int& nside, const int& L,
                                  cudaStream_t stream) {
    // Step 1: Configure kernel launch parameters
    int block_size = 128;
    int ftm_elements = 2 * L * (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    // Step 2: Launch the kernel
    spectral_extension<complex><<<grid_size, block_size, 0, stream>>>(data, output, nside, L);

    // Step 3: Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

/**
 * @brief Launches the shift/normalize CUDA kernel for HEALPix data processing.
 *
 * This function configures and launches the shift_normalize_kernel with appropriate
 * grid and block dimensions. Supports both in-place (cooperative kernel) and out-of-place
 * (regular kernel with scratch buffer) modes. For out-of-place mode with shifting, the
 * shifted data is copied back from the scratch buffer after the kernel completes.
 *
 * @tparam complex The complex type (cufftComplex or cufftDoubleComplex).
 * @param stream CUDA stream for kernel execution.
 * @param data Input/output array of HEALPix pixel data.
 * @param shift_buffer Scratch buffer for out-of-place shifting (can be nullptr for in-place).
 * @param nside The HEALPix Nside parameter.
 * @param apply_shift Flag indicating whether to apply FFT shifting.
 * @param norm Normalization type (0=by nphi, 1=by sqrt(nphi), 2=no normalization).
 * @param use_out_of_place If true, use regular launch with out-of-place buffer; if false, use cooperative
 * launch with in-place.
 * @return HRESULT indicating success or failure.
 */
template <typename complex>
HRESULT launch_shift_normalize_kernel(cudaStream_t stream, complex* data, complex* shift_buffer, int nside,
                                      bool apply_shift, int norm, bool use_out_of_place) {
    // Step 1: Configure kernel launch parameters
    long long int Npix = 12 * nside * nside;
    int block_size = 256;
    int grid_size = (Npix + block_size - 1) / block_size;

    if (use_out_of_place) {
        // Step 2a: Regular launch for out-of-place mode
        if constexpr (std::is_same_v<complex, cufftComplex>) {
            shift_normalize_kernel<cufftComplex, float><<<grid_size, block_size, 0, stream>>>(
                    (cufftComplex*)data, (cufftComplex*)shift_buffer, nside, apply_shift, norm, true);
        } else {
            shift_normalize_kernel<cufftDoubleComplex, double><<<grid_size, block_size, 0, stream>>>(
                    (cufftDoubleComplex*)data, (cufftDoubleComplex*)shift_buffer, nside, apply_shift, norm,
                    true);
        }
        checkCudaErrors(cudaGetLastError());

        // Step 2b: If shifting was applied, copy result back from scratch buffer
        if (apply_shift && shift_buffer != nullptr) {
            checkCudaErrors(cudaMemcpyAsync(data, shift_buffer, Npix * sizeof(complex),
                                            cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        // Step 3a: Set up kernel arguments for cooperative launch
        void* kernel_args[6];
        kernel_args[0] = &data;
        kernel_args[1] = &shift_buffer;
        kernel_args[2] = &nside;
        kernel_args[3] = &apply_shift;
        kernel_args[4] = &norm;
        kernel_args[5] = &use_out_of_place;

        // Step 3b: Launch cooperative kernel for in-place mode
        if constexpr (std::is_same_v<complex, cufftComplex>) {
            checkCudaErrors(cudaLaunchCooperativeKernel((void*)shift_normalize_kernel<cufftComplex, float>,
                                                        grid_size, block_size, kernel_args, 0, stream));
        } else {
            checkCudaErrors(
                    cudaLaunchCooperativeKernel((void*)shift_normalize_kernel<cufftDoubleComplex, double>,
                                                grid_size, block_size, kernel_args, 0, stream));
        }
        checkCudaErrors(cudaGetLastError());
    }

    return S_OK;
}

// ============================================================================
// C++ TEMPLATE SPECIALIZATIONS
// ============================================================================

// Explicit template specializations for spectral folding functions
template HRESULT launch_spectral_folding<cufftComplex>(cufftComplex* data, cufftComplex* output,
                                                       const int& nside, const int& L, const bool& shift,
                                                       cudaStream_t stream);
template HRESULT launch_spectral_folding<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                             cufftDoubleComplex* output, const int& nside,
                                                             const int& L, const bool& shift,
                                                             cudaStream_t stream);

// Explicit template specializations for spectral extension functions
template HRESULT launch_spectral_extension<cufftComplex>(cufftComplex* data, cufftComplex* output,
                                                         const int& nside, const int& L, cudaStream_t stream);
template HRESULT launch_spectral_extension<cufftDoubleComplex>(cufftDoubleComplex* data,
                                                               cufftDoubleComplex* output, const int& nside,
                                                               const int& L, cudaStream_t stream);

// Explicit template specializations for shift/normalize functions
template HRESULT launch_shift_normalize_kernel<cufftComplex>(cudaStream_t stream, cufftComplex* data,
                                                             cufftComplex* shift_buffer, int nside,
                                                             bool apply_shift, int norm,
                                                             bool use_out_of_place);

template HRESULT launch_shift_normalize_kernel<cufftDoubleComplex>(cudaStream_t stream,
                                                                   cufftDoubleComplex* data,
                                                                   cufftDoubleComplex* shift_buffer,
                                                                   int nside, bool apply_shift, int norm,
                                                                   bool use_out_of_place);

}  // namespace s2fftKernels
