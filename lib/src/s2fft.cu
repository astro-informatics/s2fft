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
#include <iostream>
#include <iterator>
#include <numeric>

#include <vector>
#include "s2fft_callbacks.cuh"

namespace s2fft {

HRESULT s2fftExec::Initialize(const s2fftDescriptor &descriptor, size_t &worksize) {
    m_nside = descriptor.nside;
    m_total_pixels = 12 * m_nside * m_nside;

    size_t start_index(0);
    size_t end_index(12 * m_nside * m_nside);
    size_t nphi(0);
    const bool &shift = descriptor.shift;
    m_upper_ring_offsets.reserve(m_nside - 1);
    m_lower_ring_offsets.reserve(m_nside - 1);

    for (size_t i = 0; i < m_nside - 1; i++) {
        nphi = 4 * (i + 1);
        m_upper_ring_offsets.push_back(start_index);
        m_lower_ring_offsets.push_back(end_index - nphi);

        start_index += nphi;
        end_index -= nphi;
    }
    m_equatorial_offset_start = start_index;
    m_equatorial_offset_end = end_index;
    m_equatorial_ring_num = (end_index - start_index) / (4 * m_nside);

    // Plan creation
    for (size_t i = 0; i < m_nside - 1; i++) {
        size_t polar_worksize{0};
        int64 upper_ring_offset = m_upper_ring_offsets[i];
        int64 lower_ring_offset = m_lower_ring_offsets[i];

        cufftHandle plan{};
        cufftHandle inverse_plan{};
        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftCreate(&inverse_plan));
        // Plans are done on upper and lower polar rings
        int rank = 1;                      // 1D FFT  : In our case the rank is always 1
        int batch_size = 2;                // Number of rings to transform
        int64 n[] = {4 * ((int64)i + 1)};  // Size of each FFT 4 times the ring number (first is 4, second is
                                           // 8, third is 12, etc)
        int64 inembed[] = {0};             // Stride of input data (meaningless but has to be set)
        int64 istride = 1;  // Distance between consecutive elements in the same batch always 1 since we
                            // have contiguous data
        int64 idist = lower_ring_offset -
                      upper_ring_offset;  // Distance between the starting points of two consecutive
                                          // batches, it is equal to the distance between the two rings
        int64 onembed[] = {0};            // Stride of output data (meaningless but has to be set)
        int64 ostride = 1;  // Distance between consecutive elements in the output batch, also 1 since
                            // everything is done in place
        int64 odist =
                lower_ring_offset - upper_ring_offset;  // Same as idist since we want to transform in place

        CUFFT_CALL(cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                                       CUFFT_C2C, batch_size, &polar_worksize));

        CUFFT_CALL(cufftMakePlanMany64(inverse_plan, rank, n, inembed, istride, idist, onembed, ostride,
                                       odist, CUFFT_C2C, batch_size, &polar_worksize));

        int64 params[4];
        int64 *params_dev;
        params[0] = n[0];
        params[1] = idist;
        params[2] = upper_ring_offset;
        params[3] = lower_ring_offset;
        cudaMalloc(&params_dev, 4 * sizeof(int64));
        cudaMemcpy(params_dev, params, 4 * sizeof(int64), cudaMemcpyHostToDevice);

        // Backward shift is a load call back .. set it on its own
        if (shift) {
            CUFFT_CALL(cufftXtSetCallback(inverse_plan, (void **)&s2fftKernels::ifft_shift_ptr,
                                          CUFFT_CB_LD_COMPLEX, (void **)&params_dev));
        }
        //  Set the callback for the forward and backward ffts
        switch (descriptor.norm) {
            case fft_norm::ORTHO:
                CUFFT_CALL(cufftXtSetCallback(plan, (void **)&FFT_NORM_ORTHO(shift), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
                // For inverse plans the shifting is done in the load callback
                CUFFT_CALL(cufftXtSetCallback(inverse_plan, (void **)&FFT_NORM_ORTHO(false),
                                              CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
                break;
            case fft_norm::BACKWARD:
                // No normalization is done for the forward fft but the shifting is done if requested
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(plan, (void **)&s2fftKernels::fft_shift_ptr,
                                                  CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
                }
                // For inverse plans the shifting is done in the load callback
                CUFFT_CALL(cufftXtSetCallback(inverse_plan, (void **)&FFT_NORM(false), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
                break;
            case fft_norm::FORWARD:
                CUFFT_CALL(cufftXtSetCallback(plan, (void **)&FFT_NORM(shift), CUFFT_CB_ST_COMPLEX,
                                              (void **)&params_dev));
                break;
            case fft_norm::NONE:
                if (shift) {
                    CUFFT_CALL(cufftXtSetCallback(plan, (void **)&s2fftKernels::fft_shift_ptr,
                                                  CUFFT_CB_ST_COMPLEX, (void **)&params_dev));
                }
        }

        m_polar_plans.push_back(plan);
        m_inverse_polar_plans.push_back(inverse_plan);
    }
    // Equator plan

    // Equator is a matrix with size 4 * m_nside x equatorial_ring_num
    // cufftMakePlan1d is enough for this case

    size_t equator_worksize{0};
    int64 equator_size = (4 * m_nside);
    // Forward plan
    CUFFT_CALL(cufftCreate(&m_equator_plan));
    CUFFT_CALL(cufftMakePlanMany64(m_equator_plan, 1, &equator_size, nullptr, 1, 1, nullptr, 1, 1, CUFFT_C2C,
                                   m_equatorial_ring_num, &equator_worksize));
    // Inverse plan
    CUFFT_CALL(cufftCreate(&m_inverse_equator_plan));
    CUFFT_CALL(cufftMakePlanMany64(m_inverse_equator_plan, 1, &equator_size, nullptr, 1, 1, nullptr, 1, 1,
                                   CUFFT_C2C, m_equatorial_ring_num, &equator_worksize));

    int64 equator_params[3];
    equator_params[0] = equator_size;
    equator_params[1] = m_equatorial_offset_start;
    // Dummy param, the offset of any equator element is less than the offset of the last ring
    equator_params[2] = m_equatorial_offset_end;
    int64 *equator_params_dev;
    cudaMalloc(&equator_params_dev, 3 * sizeof(int64));
    cudaMemcpy(equator_params_dev, equator_params, 3 * sizeof(int64), cudaMemcpyHostToDevice);

    if (shift) {
        CUFFT_CALL(cufftXtSetCallback(m_inverse_equator_plan, (void **)&s2fftKernels::ifft_shift_eq_ptr,
                                      CUFFT_CB_LD_COMPLEX, (void **)&equator_params_dev));
    }

    switch (descriptor.norm) {
        case fft_norm::ORTHO:
            CUFFT_CALL(cufftXtSetCallback(m_equator_plan, (void **)&FFT_NORM_ORTHO_EQ(shift),
                                          CUFFT_CB_ST_COMPLEX, (void **)&equator_params_dev));
            CUFFT_CALL(cufftXtSetCallback(m_inverse_equator_plan, (void **)&FFT_NORM_ORTHO_EQ(false),
                                          CUFFT_CB_ST_COMPLEX, (void **)&equator_params_dev));
            break;
        case fft_norm::BACKWARD:
            // No normalization is done for the forward fft but the shifting is done if requested
            if (shift)
                CUFFT_CALL(cufftXtSetCallback(m_equator_plan, (void **)&s2fftKernels::fft_shift_eq_ptr,
                                              CUFFT_CB_ST_COMPLEX, (void **)&equator_params_dev));
            // For inverse plans the shifting is done in the load callback
            CUFFT_CALL(cufftXtSetCallback(m_inverse_equator_plan, (void **)&FFT_NORM_EQ(false),
                                          CUFFT_CB_ST_COMPLEX, (void **)&equator_params_dev));
            break;
        case fft_norm::FORWARD:
            CUFFT_CALL(cufftXtSetCallback(m_equator_plan, (void **)&FFT_NORM_EQ(shift), CUFFT_CB_ST_COMPLEX,
                                          (void **)&equator_params_dev));
            break;
        case fft_norm::NONE:
            if (shift) {
                CUFFT_CALL(cufftXtSetCallback(m_equator_plan, (void **)&s2fftKernels::fft_shift_eq_ptr,
                                              CUFFT_CB_ST_COMPLEX, (void **)&equator_params_dev));
            }
    }

    return S_OK;
}

HRESULT s2fftExec::Forward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) {
    void *data_d = buffers[0];
    cufftComplex *data_c_d = static_cast<cufftComplex *>(data_d);
    // Polar rings ffts
    for (int i = 0; i < m_nside - 1; i++) {
        int upper_ring_offset = m_upper_ring_offsets[i];

        CUFFT_CALL(cufftSetStream(m_polar_plans[i], stream))
        CUFFT_CALL(cufftExecC2C(m_polar_plans[i], data_c_d + upper_ring_offset, data_c_d + upper_ring_offset,
                                CUFFT_FORWARD));
    }

    // Equator fft
    CUFFT_CALL(cufftSetStream(m_equator_plan, stream))
    CUFFT_CALL(cufftExecC2C(m_equator_plan, data_c_d + m_equatorial_offset_start,
                            data_c_d + m_equatorial_offset_start, CUFFT_FORWARD));
    //
    return S_OK;
}

HRESULT s2fftExec::Backward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) {
    void *data_d = buffers[0];
    cufftComplex *data_c_d = static_cast<cufftComplex *>(data_d);

    // Polar rings inverse FFTs
    for (int i = 0; i < m_nside - 1; i++) {
        //  if ( i > 60) continue;

        int upper_ring_offset = m_upper_ring_offsets[i];

        CUFFT_CALL(cufftSetStream(m_inverse_polar_plans[i], stream))
        CUFFT_CALL(cufftExecC2C(m_inverse_polar_plans[i], data_c_d + upper_ring_offset,
                                data_c_d + upper_ring_offset, CUFFT_INVERSE));
    }
    // Equator inverse FFT
    CUFFT_CALL(cufftSetStream(m_inverse_equator_plan, stream))
    CUFFT_CALL(cufftExecC2C(m_inverse_equator_plan, data_c_d + m_equatorial_offset_start,
                            data_c_d + m_equatorial_offset_start, CUFFT_INVERSE));
    //
    return S_OK;
}
}  // namespace s2fft