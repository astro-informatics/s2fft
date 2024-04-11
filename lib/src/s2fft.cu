#include "s2fft.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <algorithm>
#include <iostream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include <vector>
#include "cufft.h"

#define TILE_SIZE 256

namespace s2fft {

__global__ void s2fft_rings_2_nphi_kernel(float *healpixArray, int nside) {
    int numRings = 2 * nside - 1;
    int numPixelsInRing = 4 * (numRings - 1);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numPixelsInRing) {
        int ringIdx = tid / (4 * nside);
        int phiIdx = tid % (4 * nside);

        int ringSize = 4 * (ringIdx + 1);
        int pixelIdx;
        if (phiIdx < 2 * nside) {
            // Forward
            pixelIdx = ringSize * (phiIdx % nside) + phiIdx / nside;
        } else {
            // Backward
            phiIdx -= 2 * nside;
            pixelIdx = ringSize * (phiIdx % nside) + (phiIdx / nside) + (4 * nside - 1 - ringSize);
        }

        // Swap pixels in-place
        int temp = healpixArray[tid];
        healpixArray[tid] = healpixArray[pixelIdx];
        healpixArray[pixelIdx] = temp;
    }
}

void s2fft_rings_2_nphi(float *data, int nside) {
    int numRings = 2 * nside - 1;
    int numPixelsInRing = 4 * (numRings - 1);

    int numBlocks = ceil(numPixelsInRing / (float)TILE_SIZE);
    s2fft_rings_2_nphi_kernel<<<numBlocks, TILE_SIZE>>>(data, nside);
}

HRESULT s2fftExec::Initialize(const s2fftDescriptor &descriptor, size_t &worksize) {

    const int &nside = descriptor.nside;

    std::vector<int> polar_offsets(nside - 1);
    int start_index = 0;
    for (int t = 0; t < nside - 1; t++) {
        int nphi = 4 * (t + 1);
        polar_offsets[t] = start_index;
        start_index += nphi;
    }
    std::cout << "size of polar_offsets: " << polar_offsets.size() << std::endl;

    // print(polar_offsets)
    std::copy(polar_offsets.begin(), polar_offsets.end(), std::ostream_iterator<int>(std::cout, " "));

    std::cout << std::endl << std::endl;

    return S_OK;
}

HRESULT s2fftExec::Forward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) { return S_OK; }

HRESULT s2fftExec::Backward(const s2fftDescriptor &desc, cudaStream_t stream, void **buffers) { return S_OK; }

}  // namespace s2fft