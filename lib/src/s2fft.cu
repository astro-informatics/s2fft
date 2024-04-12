#include "s2fft.h"
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

#include <vector>
#include "cufft.h"
#include <cufftXt.h>

namespace s2fft {

//__device__ cufftComplex normalize_cb(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
//    cufftComplex *data = (cufftComplex *)dataIn;
//    cufftComplex result;
//}

HRESULT s2fftExec::Initialize(const s2fftDescriptor &descriptor, size_t &worksize) {
    m_nside = descriptor.nside;
    m_total_pixels = 12 * m_nside * m_nside;

    int start_index(0);
    int end_index(12 * m_nside * m_nside);
    int nphi(0);

    for (int i = 0; i < m_nside - 1; i++) {
        nphi = 4 * (i + 1);
        m_upper_ring_offsets.push_back(start_index);
        m_lower_ring_offsets.push_back(end_index - nphi);

        start_index += nphi;
        end_index -= nphi;
    }
    equatorial_offset = start_index;
    equatorial_ring_num = (end_index - start_index) / (4 * m_nside);

    // Plan creation
    for (int i = 0; i < m_nside - 1; i++) {
        size_t polar_worksize{0};
        int upper_ring_offset = m_upper_ring_offsets[i];
        int lower_ring_offset = m_lower_ring_offsets[i];

        cufftHandle plan{};
        cufftHandle inverse_plan{};
        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftCreate(&inverse_plan));
        // Plans are done on upper and lower polar rings

        // cufftResult cufftPlanMany(cufftHandle * plan, int rank, int *n, int *inembed, int istride, int
        // idist,
        //                           int *onembed, int ostride, int odist, cufftType type, int batch);

        // rank : In our case the rank is always 1
        // the size is 4 times the ring number (first is 4, second is 8, third is 12, etc)
        // inembed and onembed are always NULL for 1D transforms
        // istride and ostride are always 1 because the data is contiguous
        // idist and odist are the distance between the two rings
        // batch is always 2 because we have two rings to transform

        int rank = 1;             // 1D FFT
        int batch_size = 2;       // Number of rings to transform
        int n[] = {4 * (i + 1)};  // Size of each FFT
        int inembed[] = {0};      // Stride of input data (meaningless but has to be set)
        int istride = 1;          // Distance between consecutive elements in the same batch
        int idist = lower_ring_offset -
                    upper_ring_offset;  // Distance between the starting points of two consecutive batches
        int onembed[] = {0};            // Stride of output data (meaningless but has to be set)
        int ostride = 1;                // Distance between consecutive elements in the same batch
        int odist = lower_ring_offset - upper_ring_offset;

        if (i == 11 || i == 12) std::cout << "polar_worksize: " << polar_worksize << std::endl;

        CUFFT_CALL(cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                                     CUFFT_C2C, batch_size, &polar_worksize));
        // CUFFT_CALL(cufftMakePlanMany(inverse_plan, rank, n, inembed, istride, idist, onembed, ostride,
        // odist,
        //                              CUFFT_C2C, batch_size, &polar_worksize));

        m_polar_plans.push_back(plan);

    }
    // Equator plan

    // Equator is a matrix with size 4 * m_nside x equatorial_ring_num
    // cufftMakePlan1d is enough for this case

    size_t equator_worksize{0};
    CUFFT_CALL(cufftCreate(&m_equator_plan));
    CUFFT_CALL(cufftMakePlan1d(m_equator_plan, (4 * m_nside), CUFFT_C2C, equatorial_ring_num,
                               &equator_worksize));

    return S_OK;
}

HRESULT s2fftExec::Forward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) {
    void *data_d = buffers[0];
    cufftComplex *data_c_d = static_cast<cufftComplex *>(data_d);

    // Polar rings ffts
    std::cout << "number of plans: " << m_polar_plans.size() << std::endl;
    for (int i = 0; i < m_nside - 1; i++) {
        int upper_ring_offset = m_upper_ring_offsets[i];

        CUFFT_CALL(cufftSetStream(m_polar_plans[i], stream))
        CUFFT_CALL(cufftExecC2C(m_polar_plans[i], data_c_d + upper_ring_offset, data_c_d + upper_ring_offset,
                                CUFFT_FORWARD));

    }

    // Equator fft
    CUFFT_CALL(cufftSetStream(m_equator_plan, stream))
    CUFFT_CALL(cufftExecC2C(m_equator_plan, data_c_d + equatorial_offset, data_c_d + equatorial_offset,
                           CUFFT_FORWARD));

    return S_OK;
}

HRESULT s2fftExec::Backward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) {
    void *data_d = buffers[0];
    cufftComplex *data_c_d = static_cast<cufftComplex *>(data_d);

    // Polar rings inverse FFTs
    for (int i = 0; i < m_nside - 1; i++) {
        int upper_ring_offset = m_upper_ring_offsets[i];

        CUFFT_CALL(cufftSetStream(m_polar_plans[i], stream))
        CUFFT_CALL(cufftExecC2C(m_polar_plans[i], data_c_d + upper_ring_offset, data_c_d + upper_ring_offset,
                                CUFFT_INVERSE));
    }

    // Equator inverse FFT
    CUFFT_CALL(cufftSetStream(m_equator_plan, stream))
    CUFFT_CALL(cufftExecC2C(m_equator_plan, data_c_d + equatorial_offset, data_c_d + equatorial_offset,
                            CUFFT_INVERSE));

    return S_OK;
}
}  // namespace s2fft