#include <algorithm>
#include "s2fft.h"
#include <array>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

using namespace s2fft;

int main() {
    int nside = 4;
    int L = 2 * nside;
    int total_pixels = 12 * nside * nside;
    int number_of_rings = 2 * nside - 1;

    thrust::host_vector<float> h_vec(total_pixels);
    thrust::sequence(h_vec.begin(), h_vec.end());
    int ring_offset = 0;

    for (int i = 0; i < (nside * 2) + 1; i++) {
        int nphi = 4 * (i < nside ? i : 2 * nside - i);
        std::copy(h_vec.begin() + ring_offset, h_vec.begin() + ring_offset + nphi,
                  std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        ring_offset += nphi;
    }

    std::cout << std::endl;

    thrust::device_vector<float> d_vec = h_vec;

    s2fft_rings_2_nphi(thrust::raw_pointer_cast(d_vec.data()), nside);

    thrust::host_vector<float> h_vec2 = d_vec;

    ring_offset = 0;
    for (int i = 0; i < nside - 1; i++) {
        int nphi =  4 * (i + 1);
        std::copy(h_vec2.begin() + ring_offset, h_vec2.begin() + ring_offset + nphi,
                  std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        ring_offset += nphi;
        std::copy(h_vec2.begin() + ring_offset, h_vec2.begin() + ring_offset + nphi,
                  std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        ring_offset += nphi;
    }
    // Print equatorial ring
    int nphi = 4 * nside;
    std::copy(h_vec2.begin() + ring_offset, h_vec2.begin() + ring_offset + nphi,
              std::ostream_iterator<float>(std::cout, " "));

    std::cout << std::endl;

    std::cout << "nsides: " << nside << std::endl;
    std::cout << "total_pixels: " << total_pixels << std::endl;
    std::cout << "number_of_rings: " << number_of_rings << std::endl;

    s2fftDescriptor desc(nside, L, true, true);
    s2fftExec exec;
    size_t worksize;
    // exec.Initialize(desc, worksize);

    // Allocate memory on cpu_data
}
