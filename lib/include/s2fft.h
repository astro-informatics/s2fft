
#ifndef S2FFT_H
#define S2FFT_H

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <iostream>
#include <vector>
#include "cufft.h"


namespace s2fft {

class s2fftDescriptor {
public:
    int nside;
    int total_pixels;
    int number_of_rings;
    std::vector<int> ring_offsets;

    s2fftDescriptor(int nside);
    s2fftDescriptor() = default;
    ~s2fftDescriptor() = default;

    bool operator==(const s2fftDescriptor &other) const;
};

class s2fftExec {
    using complex_t = cuda::std::complex<float>;
    friend class PlanCache;

public:
    s2fftExec() {}
    ~s2fftExec() {}

    HRESULT Initialize(const int &nside, const int &harmonic_band_limit, const bool &reality);

    HRESULT Forward(const complex_t *input, complex_t *output);

    HRESULT Backward(const complex_t *input, complex_t *output);
};

}  // namespace s2fft

#endif  // S2FFT_H