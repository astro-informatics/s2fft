
#ifndef S2FFT_H
#define S2FFT_H

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <iostream>
#include <vector>
#include "cufft.h"

namespace s2fft {

void s2fft_rings_2_nphi(float* data, int nside);

void s2fft_nphi_2_rings(float* data, int nside);


class s2fftDescriptor {
public:
    int nside;
    int harmonic_band_limit;
    bool reality;
    bool forward = true;

    s2fftDescriptor(int nside, int harmonic_band_limit, bool reality, bool forward)
            : nside(nside), harmonic_band_limit(harmonic_band_limit), reality(reality), forward(forward) {}
    s2fftDescriptor() = default;
    ~s2fftDescriptor() = default;

    bool operator==(const s2fftDescriptor &other) const {
        return nside == other.nside && harmonic_band_limit == other.harmonic_band_limit &&
               reality == other.reality;
    }
};

class s2fftExec {
    using complex_t = cuda::std::complex<float>;
    friend class PlanCache;

public:
    s2fftExec() {}
    ~s2fftExec() {}

    HRESULT Initialize(const s2fftDescriptor &descriptor, size_t &worksize);

    HRESULT Forward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers);

    HRESULT Backward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers);

private:
    std::vector<cufftHandle> m_polar_plans;
    cufftHandle m_equator_plan;
    std::vector<cufftHandle> m_inverse_polar_plans;
    cufftHandle m_inverse_equator_plan;
    int m_nside;
    int m_total_pixels;
    int equatorial_offset;
    int equatorial_ring_num;
    std::vector<int> m_upper_ring_offsets;
    std::vector<int> m_lower_ring_offsets;
};

}  // namespace s2fft

namespace std {
template <>
struct hash<s2fft::s2fftDescriptor> {
    std::size_t operator()(const s2fft::s2fftDescriptor &k) const {
        size_t hash = std::hash<int>()(k.nside) ^ (std::hash<int>()(k.harmonic_band_limit) << 1) ^
                      (std::hash<bool>()(k.reality) << 2);
        return hash;
    }
};
}  // namespace std

#endif  // S2FFT_H