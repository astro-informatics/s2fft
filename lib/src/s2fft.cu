#include "s2fft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>

#include <vector>
#include "s2fft_kernels.h"

namespace s2fft {

template <typename Complex>
HRESULT s2fftExec<Complex>::Initialize(const s2fftDescriptor &descriptor) {
    // Step 1: Store the Nside parameter from the descriptor.
    m_nside = descriptor.nside;

    // Step 2: Initialize variables for ring offsets and workspace size.
    size_t start_index(0);
    size_t end_index(12 * m_nside * m_nside);
    size_t nphi(0);
    size_t worksize(0);
    // Step 3: Determine the cuFFT C2C type based on the complex type.
    const cufftType C2C_TYPE = get_cufft_type_c2c(Complex({0.0, 0.0}));
    // Step 4: Reserve space for upper and lower ring offset vectors.
    m_upper_ring_offsets.reserve(m_nside - 1);
    m_lower_ring_offsets.reserve(m_nside - 1);

    // Step 5: Calculate and store offsets for polar rings.
    for (size_t i = 0; i < m_nside - 1; i++) {
        nphi = 4 * (i + 1);
        m_upper_ring_offsets.push_back(start_index);
        m_lower_ring_offsets.push_back(end_index - nphi);

        start_index += nphi;
        end_index -= nphi;
    }  //
    // Step 6: Store offsets and number of equatorial rings.
    m_equatorial_offset_start = start_index;
    m_equatorial_offset_end = end_index;
    m_equatorial_ring_num = (end_index - start_index) / (4 * m_nside);

    // Step 7: Create cuFFT plans for polar rings.
    for (size_t i = 0; i < m_nside - 1; i++) {
        size_t polar_worksize{0};
        int64 upper_ring_offset = m_upper_ring_offsets[i];
        int64 lower_ring_offset = m_lower_ring_offsets[i];

        // Step 7a: Create cuFFT handles for forward and inverse plans.
        cufftHandle plan{};
        cufftHandle inverse_plan{};
        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftCreate(&inverse_plan));

        // Step 7b: Define parameters for 1D FFTs on polar rings.
        int rank = 1;                      // 1D FFT
        int batch_size = 2;                // Number of rings to transform (upper and lower)
        int64 n[] = {4 * ((int64)i + 1)};  // Size of each FFT
        int64 inembed[] = {0};             // Stride of input data (meaningless but has to be set)
        int64 istride = 1;                 // Distance between consecutive elements in the same batch
        int64 idist = lower_ring_offset -
                      upper_ring_offset;  // Distance between starting points of two consecutive batches
        int64 onembed[] = {0};            // Stride of output data (meaningless but has to be set)
        int64 ostride = 1;                // Distance between consecutive elements in the output batch
        int64 odist = lower_ring_offset - upper_ring_offset;  // Same as idist for in-place transform

        // Step 7c: Create cuFFT plans for forward and inverse polar transforms.
        CUFFT_CALL(cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                                       C2C_TYPE, batch_size, &polar_worksize));
        // Step 7d: Update overall maximum workspace size.
        worksize = std::max(worksize, polar_worksize);

        CUFFT_CALL(cufftMakePlanMany64(inverse_plan, rank, n, inembed, istride, idist, onembed, ostride,
                                       odist, C2C_TYPE, batch_size, &polar_worksize));
        // Step 7e: Update overall maximum workspace size again.
        worksize = std::max(worksize, polar_worksize);

        // Step 7f: Store the created plans.
        m_polar_plans.push_back(plan);
        m_inverse_polar_plans.push_back(inverse_plan);
    }

    // Step 8: Create cuFFT plans for the equatorial ring.
    size_t equator_worksize{0};
    int64 equator_size = (4 * m_nside);

    // Step 8a: Create cuFFT handle for the forward equatorial plan.
    CUFFT_CALL(cufftCreate(&m_equator_plan));
    CUFFT_CALL(cufftMakePlanMany64(m_equator_plan, 1, &equator_size, nullptr, 1, 1, nullptr, 1, 1, C2C_TYPE,
                                   m_equatorial_ring_num, &equator_worksize));
    // Step 8b: Update overall maximum workspace size.
    worksize = std::max(worksize, equator_worksize);

    // Step 8c: Create cuFFT handle for the inverse equatorial plan.
    CUFFT_CALL(cufftCreate(&m_inverse_equator_plan));
    CUFFT_CALL(cufftMakePlanMany64(m_inverse_equator_plan, 1, &equator_size, nullptr, 1, 1, nullptr, 1, 1,
                                   C2C_TYPE, m_equatorial_ring_num, &equator_worksize));
    // Step 8d: Update overall maximum workspace size again.
    worksize = std::max(worksize, equator_worksize);
    // Step 9: Store the final maximum workspace size.
    this->m_work_size = worksize;

    return S_OK;
}

template <typename Complex>
HRESULT s2fftExec<Complex>::Forward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data,
                                    Complex *workspace, Complex *shift_scratch, bool use_out_of_place) {
    // Step 1: Determine the FFT direction (forward or inverse based on adjoint flag).
    const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;
    // Step 2: Extract normalization, shift, and double precision flags from the descriptor.
    const s2fftKernels::fft_norm &norm = desc.norm;
    const bool &shift = desc.shift;

    // Step 3: Execute FFTs for polar rings.
    for (int i = 0; i < m_nside - 1; i++) {
        // Step 3a: Get upper and lower ring offsets.
        int upper_ring_offset = m_upper_ring_offsets[i];

        // Step 3e: Set the CUDA stream and work area for the cuFFT plan.
        CUFFT_CALL(cufftSetStream(m_polar_plans[i], stream));
        CUFFT_CALL(cufftSetWorkArea(m_polar_plans[i], workspace));
        // Step 3f: Execute the cuFFT transform.
        CUFFT_CALL(
                cufftXtExec(m_polar_plans[i], data + upper_ring_offset, data + upper_ring_offset, DIRECTION));
    }
    // Step 4: Execute FFT for the equatorial ring.
    // Step 4d: Set the CUDA stream and work area for the equatorial cuFFT plan.
    CUFFT_CALL(cufftSetStream(m_equator_plan, stream));
    CUFFT_CALL(cufftSetWorkArea(m_equator_plan, workspace));
    // Step 4e: Execute the cuFFT transform for the equator.
    CUFFT_CALL(cufftXtExec(m_equator_plan, data + m_equatorial_offset_start, data + m_equatorial_offset_start,
                           DIRECTION));

    // Step 5: Launch the custom kernel for normalization and shifting.
    switch (norm) {
        case s2fftKernels::fft_norm::NONE:
        case s2fftKernels::fft_norm::BACKWARD:
            // No normalization, only shift if required.
            s2fftKernels::launch_shift_normalize_kernel(stream, data, shift_scratch, m_nside, shift, 2,
                                                        use_out_of_place);
            break;
        case s2fftKernels::fft_norm::FORWARD:
            // Normalize by sqrt(Npix).
            s2fftKernels::launch_shift_normalize_kernel(stream, data, shift_scratch, m_nside, shift, 0,
                                                        use_out_of_place);
            break;
        case s2fftKernels::fft_norm::ORTHO:
            // Normalize by Npix.
            s2fftKernels::launch_shift_normalize_kernel(stream, data, shift_scratch, m_nside, shift, 1,
                                                        use_out_of_place);
            break;
        default:
            return E_INVALIDARG;  // Invalid normalization type.
    }

    return S_OK;
}

template <typename Complex>
HRESULT s2fftExec<Complex>::Backward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data,
                                     Complex *workspace, Complex *shift_scratch, bool use_out_of_place) {
    // Step 1: Determine the FFT direction (forward or inverse based on adjoint flag).
    const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;
    // Step 2: Extract normalization, shift, and double precision flags from the descriptor.
    const s2fftKernels::fft_norm &norm = desc.norm;

    // Step 3: Execute inverse FFTs for polar rings.
    for (int i = 0; i < m_nside - 1; i++) {
        // Step 3a: Get upper and lower ring offsets.
        int upper_ring_offset = m_upper_ring_offsets[i];

        // Step 3e: Set the CUDA stream and work area for the cuFFT plan.
        CUFFT_CALL(cufftSetStream(m_inverse_polar_plans[i], stream));
        CUFFT_CALL(cufftSetWorkArea(m_inverse_polar_plans[i], workspace));
        // Step 3f: Execute the cuFFT transform.
        CUFFT_CALL(cufftXtExec(m_inverse_polar_plans[i], data + upper_ring_offset, data + upper_ring_offset,
                               DIRECTION));
    }
    // Step 4: Execute inverse FFT for the equatorial ring.
    // Step 4d: Set the CUDA stream and work area for the equatorial cuFFT plan.
    CUFFT_CALL(cufftSetStream(m_inverse_equator_plan, stream));
    CUFFT_CALL(cufftSetWorkArea(m_inverse_equator_plan, workspace));
    // Step 4e: Execute the cuFFT transform for the equator.
    CUFFT_CALL(cufftXtExec(m_inverse_equator_plan, data + m_equatorial_offset_start,
                           data + m_equatorial_offset_start, DIRECTION));

    // Step 5: Launch the custom kernel for normalization and shifting.
    switch (norm) {
        case s2fftKernels::fft_norm::NONE:
        case s2fftKernels::fft_norm::FORWARD:
            // No normalization, do nothing.
            break;
        case s2fftKernels::fft_norm::BACKWARD:
            // Normalize by sqrt(Npix).
            s2fftKernels::launch_shift_normalize_kernel(stream, data, shift_scratch, m_nside, false, 0,
                                                        use_out_of_place);
            break;
        case s2fftKernels::fft_norm::ORTHO:
            // Normalize by Npix.
            s2fftKernels::launch_shift_normalize_kernel(stream, data, shift_scratch, m_nside, false, 1,
                                                        use_out_of_place);
            break;
        default:
            return E_INVALIDARG;  // Invalid normalization type.
    }

    return S_OK;
}

template class s2fftExec<cufftComplex>;
template class s2fftExec<cufftDoubleComplex>;

}  // namespace s2fft