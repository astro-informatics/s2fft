
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
#include "thrust/device_vector.h"

namespace s2fft {

enum fft_norm { FORWARD = 1, BACKWARD = 2, ORTHO = 3, NONE = 4 };

void s2fft_rings_2_nphi(float *data, int nside);

void s2fft_nphi_2_rings(float *data, int nside);

typedef struct _cb_params {
    int norm;
    int direction;
    bool shift;
    int size;
} cb_params;

class s2fftDescriptor {
public:
    int nside;
    int harmonic_band_limit;
    bool reality;

    bool forward = true;
    fft_norm norm = BACKWARD;
    bool shift = true;

    s2fftDescriptor(int nside, int harmonic_band_limit, bool reality, bool forward = true,
                    fft_norm norm = BACKWARD, bool shift = true)
            : nside(nside),
              harmonic_band_limit(harmonic_band_limit),
              reality(reality),
              norm(norm),
              forward(forward),
              shift(shift) {}

    s2fftDescriptor() = default;
    ~s2fftDescriptor() = default;

    bool operator==(const s2fftDescriptor &other) const {
        return nside == other.nside && harmonic_band_limit == other.harmonic_band_limit &&
               reality == other.reality && norm == other.norm && forward == other.forward &&
               shift == other.shift;
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
    int m_equatorial_offset;
    int equatorial_ring_num;
    std::vector<int> m_upper_ring_offsets;
    std::vector<int> m_lower_ring_offsets;

    // Callback params stored for cleanup purposes
    thrust::device_vector<cb_params> m_cb_params;
};

}  // namespace s2fft

namespace std {
template <>
struct hash<s2fft::s2fftDescriptor> {
    std::size_t operator()(const s2fft::s2fftDescriptor &k) const {
        size_t hash = std::hash<int>()(k.nside) ^ (std::hash<int>()(k.harmonic_band_limit) << 1) ^
                      (std::hash<bool>()(k.reality) << 2) ^ (std::hash<int>()(k.norm) << 3) ^
                      (std::hash<bool>()(k.shift) << 4);
        return hash;
    }
};
}  // namespace std

#endif  // S2FFT_H