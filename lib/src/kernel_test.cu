#include <algorithm>
#include "s2fft.h"
#include <array>
#include <iostream>
#include <ostream>
#include <sstream>
#include "cufft.h"
#include "perfostep.hpp"
#include "s2fft_kernels.h"
#include <argparse.hpp>

using namespace s2fft;

enum FFTType { FORWARD, BACKWARD, BOTH, NONE };

using Type = cuComplex;

void run_test(int nside) {
    int L = 2 * nside;
    int total_pixels = 12 * nside * nside;

    // Compute the flm size
    int polar_pixels = 4 * nside * (nside - 1);
    int equator_rings_num = (total_pixels - polar_pixels) / (4 * nside);
    int num_rings = equator_rings_num + 2 * (nside - 1);
    int flm_size = num_rings * (4 * nside);

    cuComplex *h_vec_in = new cuComplex[total_pixels];
    cuComplex *h_vec_out = new cuComplex[total_pixels];
    cuComplex *d_vec_in;
    cuComplex *d_vec_out;

    cudaMalloc(&d_vec_in, total_pixels * sizeof(cuComplex));
    cudaMalloc(&d_vec_out, flm_size * sizeof(cuComplex));

    // Initialize host vectors using std::generate
    int start_index(0);
    std::generate(h_vec_in, h_vec_in + total_pixels, [&start_index]() {
        cuComplex c;
        c.x = start_index;
        c.y = 0.0f;
        start_index += 1;
        return c;
    });

    ////// Create cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    s2fftDescriptor desc(nside, L, true, true, fft_norm::BACKWARD, false);
    s2fftExec exec;
    size_t worksize(0);
    exec.Initialize(desc, worksize);

    s2fftKernels::launch_spectral_extension(d_vec_in, d_vec_out, nside
    , L, exec.m_equatorial_offset_start, exec.m_equatorial_offset_end, stream);
    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaGetLastError());

    // Copy device data to host
    //cudaMemcpy(h_vec_out, d_vec_out, flm_size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < flm_size; i++) {
        std::cout << "[" << i << "] " << h_vec_out[i].x << " + " << h_vec_out[i].y << "i" << std::endl;
    }

    // Free memory
    delete[] h_vec_in;
    delete[] h_vec_out;
    cudaFree(d_vec_in);
    cudaFree(d_vec_out);
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("s2fft_kernels_test");
    program.add_argument("-s", "--nside").help("nside").scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(1);
    }
    int nside = program.get<int>("--nside");
    // bool test = program.get<bool>("--check");
    // bool print_res = program.get<bool>("--print");

    run_test(nside);

    return 0;
}
