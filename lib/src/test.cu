#include <algorithm>
#include "s2fft.h"
#include <array>
#include <iostream>
#include <ostream>
#include <sstream>
#include "cufft.h"
#include "perfostep.hpp"

using namespace s2fft;

using Type = cuComplex;

int main() {
    Perfostep perfostep;
    int nside = 4;
    int L = 2 * nside;
    int total_pixels = 12 * nside * nside;


    // Compute the flm size
    int polar_pixels = 4 * nside * (nside - 1);
    int equator_rings_num = (total_pixels - polar_pixels) / (4 * nside);
    int num_rings = equator_rings_num + 2 * (nside - 1);
    int flm_size = num_rings * (4 * nside);


    std::cout << "Total pixels: " << total_pixels << std::endl;

    cuComplex *h_vec_in = new cuComplex[total_pixels];
    cuComplex *h_vec_out = new cuComplex[total_pixels];
    cuComplex *d_vec;
    cuComplex *d_vec_out;
    
    cudaMallocManaged(&d_vec, total_pixels * sizeof(cuComplex));
    cudaMallocManaged(&d_vec_out, flm_size * sizeof(cuComplex));

    // Initialize host vectors using std::generate
    int start_index(0);
    std::generate(h_vec_in, h_vec_in + total_pixels, [&start_index]() {
        cuComplex c;
        c.x = start_index;
        c.y = 0.0f;
        start_index += 1;
        return c;
    });

    // Generate sequence

    // Copy host data to device
    cudaMemcpy(d_vec, h_vec_in, total_pixels * sizeof(cuComplex), cudaMemcpyHostToDevice);

    std::cout << "Original" << std::endl;
    for (int i = 0; i < total_pixels; i++) {
        std::cout << "[" << i << "] " << h_vec_in[i].x << " + " << h_vec_in[i].y << "i" << std::endl;

        if (i == 4) break;
    }

    s2fftDescriptor desc(nside, L, true, true);
    s2fftExec exec;
    size_t worksize(0);
    perfostep.Start("Initialize");
    exec.Initialize(desc, worksize);
    perfostep.Stop();
    std::cout << "worksize: " << worksize << std::endl;

    //
    //// Create cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //// Set first buffer to the data
    void **buffers = (void **)malloc(2 * sizeof(void *));
    buffers[0] = d_vec;
    worksize = total_pixels * sizeof(cuComplex);
    buffers[1] = (void *)worksize;
    buffers[2] = d_vec_out;

    //// ********************************************************
    //// Perform forward
    //// ********************************************************
    perfostep.Start("Forward");
    exec.Forward(desc, stream, buffers);
    cudaStreamSynchronize(stream);
    perfostep.Stop();
    std::cout << "Executed Forward" << std::endl;

    // ********************************************************
    // Perform Backward
    // ********************************************************
    perfostep.Start("Backward");
    exec.Backward(desc, stream, buffers);
    cudaStreamSynchronize(stream);
    perfostep.Stop();
    std::cout << "Executed Backward" << std::endl;

    perfostep.Report();

    // Copy device data to host
    cudaMemcpy(h_vec_out, d_vec, total_pixels * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    std::cout << "Output" << std::endl;
    for (int i = 0; i < total_pixels; i++) {
        std::cout << "[" << i << "] " << h_vec_out[i].x << " + " << h_vec_out[i].y << "i" << std::endl;

        if (i == 4) break;
    }

    std::cout << "Done" << std::endl;
    return 0;
}
