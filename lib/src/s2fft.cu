#include "s2fft.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <iostream>
#include <vector>
#include "cufft.h"


namespace s2fft {

HRESULT s2fftExec::Initialize(const int &nside, const int &harmonic_band_limit, const bool &reality) {
    

    //ftm_rows_polar = []
    //start_index, end_index = 0, 12 * nside**2
    //for t in range(0, nside - 1):
    //    nphi = 4 * (t + 1)
    //    f_chunks = jnp.stack(
    //        (f[start_index : start_index + nphi], f[end_index - nphi : end_index])
    //    )
    //    ftm_rows_polar.append(f_chunks_to_ftm_rows(f_chunks, nphi))
    //    start_index, end_index = start_index + nphi, end_index - nphi
    //ftm_rows_polar = jnp.stack(ftm_rows_polar)

    std::vector<int> polar_offsets;
    int start_index = 0;
    int end_index = 12 * nside * nside;
    

}

HRESULT s2fftExec::Forward(const complex_t *input, complex_t *output) { return S_OK; }

HRESULT s2fftExec::Backward(const complex_t *input, complex_t *output) { return S_OK; }

}  // namespace s2fft