#ifndef _S2FFT_KERNELS_H
#define _S2FFT_KERNELS_H


namespace s2fftKernels{
    
    void arrangeRings(float* d_data, int nside, int* d_ring_offsets, cudaStream_t stream);
}








#endif  // _S2FFT_KERNELS_H