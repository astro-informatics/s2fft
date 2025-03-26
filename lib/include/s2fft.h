
#ifndef S2FFT_H
#define S2FFT_H

#define FORWARD_NORM 1
#define BACKWARD_NORM 2
#define ORTHO_NORM 3
#define NONE_NORM 4

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <iostream>
#include <vector>
#include "cufft.h"
#include "cufftXt.h"
#include "thrust/device_vector.h"
#include "s2fft_callbacks.h"

namespace s2fft {

static cufftType get_cufft_type_c2c(cufftDoubleComplex) { return CUFFT_Z2Z; }
static cufftType get_cufft_type_c2c(cufftComplex) { return CUFFT_C2C; }

void s2fft_rings_2_nphi(float *data, int nside);

void s2fft_nphi_2_rings(float *data, int nside);

class s2fftDescriptor {
public:
    int64_t nside;
    int64_t harmonic_band_limit;
    bool reality;

    bool forward = true;
    s2fftKernels::fft_norm norm = s2fftKernels::BACKWARD;
    bool shift = true;
    bool double_precision = false;

    s2fftDescriptor(int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward = true,
                    s2fftKernels::fft_norm norm = s2fftKernels::BACKWARD, bool shift = true,
                    bool double_precision = false)
            : nside(nside),
              harmonic_band_limit(harmonic_band_limit),
              reality(reality),
              norm(norm),
              forward(forward),
              shift(shift),
              double_precision(double_precision) {}

    s2fftDescriptor() = default;
    ~s2fftDescriptor() = default;

    bool operator==(const s2fftDescriptor &other) const {
        return nside == other.nside && harmonic_band_limit == other.harmonic_band_limit &&
               reality == other.reality && norm == other.norm && shift == other.shift &&
               double_precision == other.double_precision;
    }
};

template <typename Complex>
class s2fftExec {
    friend class PlanCache;

public:
    s2fftExec() {}
    ~s2fftExec() {}

    HRESULT Initialize(const s2fftDescriptor &descriptor, size_t &worksize);

    HRESULT Forward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data);

    HRESULT Backward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data);

public:
    std::vector<cufftHandle> m_polar_plans;
    cufftHandle m_equator_plan;
    std::vector<cufftHandle> m_inverse_polar_plans;
    cufftHandle m_inverse_equator_plan;
    int m_nside;
    int m_equatorial_ring_num;
    int64 m_total_pixels;
    int64 m_equatorial_offset_start;
    int64 m_equatorial_offset_end;
    std::vector<int64> m_upper_ring_offsets;
    std::vector<int64> m_lower_ring_offsets;

    // Callback params stored for cleanup purposes
    // thrust::device_vector<cb_params> m_cb_params;
};

}  // namespace s2fft

namespace std {
template <>
struct hash<s2fft::s2fftDescriptor> {
    std::size_t operator()(const s2fft::s2fftDescriptor &k) const {
        size_t hash = std::hash<int64_t>()(k.nside) ^ (std::hash<int64_t>()(k.harmonic_band_limit) << 1) ^
                      (std::hash<bool>()(k.reality) << 2) ^ (std::hash<int>()(k.norm) << 3) ^
                      (std::hash<bool>()(k.shift) << 4) ^ (std::hash<bool>()(k.double_precision) << 5);
        return hash;
    }
};
}  // namespace std

#endif  // S2FFT_H