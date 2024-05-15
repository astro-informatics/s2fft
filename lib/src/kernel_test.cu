#include <algorithm>
#include "s2fft.h"
#include <array>
#include <complex>
#include <iostream>
#include <ostream>
#include <sstream>
#include "cufft.h"
#include "perfostep.hpp"
#include "s2fft_kernels.h"
#include <argparse.hpp>

using namespace s2fft;

enum FFTType { FORWARD, BACKWARD, BOTH, NONE };

void run_test(int nside, std::string type, int L, bool print_res) {
    int total_pixels = 12 * nside * nside;

    // Compute the flm size
    int polar_pixels = 4 * nside * (nside - 1);
    int equator_rings_num = (total_pixels - polar_pixels) / (4 * nside);
    int num_rings = 4 * nside - 1;
    int flm_size = num_rings * (2 * L);

    int *h_vec_in = new int[total_pixels];
    int *h_vec_out = new int[flm_size];
    int *d_vec_in;
    int *d_vec_out;

    cudaMalloc(&d_vec_in, total_pixels * sizeof(int));
    cudaMalloc(&d_vec_out, flm_size * sizeof(int));

    // Initialize host vectors using std::generate
    int start_index(0);
    std::generate(h_vec_in, h_vec_in + total_pixels, [&start_index]() {
        int c;
        c = start_index;
        start_index += 1;
        return c;
    });

    // Copy host data to device
    checkCudaErrors(cudaMemcpy(d_vec_in, h_vec_in, total_pixels * sizeof(int), cudaMemcpyHostToDevice));

    ////// Create cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    s2fftDescriptor desc(nside, L, true, true, fft_norm::BACKWARD, false);
    s2fftExec exec;
    size_t worksize(0);
    exec.Initialize(desc, worksize);

    s2fftKernels::launch_spectral_extension(d_vec_in, d_vec_out, nside, L, exec.m_equatorial_offset_start,
                                            exec.m_equatorial_offset_end, stream);
    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaGetLastError());

    // Copy device data to host
    checkCudaErrors(cudaMemcpy(h_vec_out, d_vec_out, flm_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < flm_size; i++) {
        std::cout << "[" << i << "] " << h_vec_out[i] << std::endl;
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
    program.add_argument("-t", "--type")
            .help("Spectral type : extended folded or both")
            .default_value("extended");
    program.add_argument("-L", "--lmax").help("lmax").default_value(2).scan<'i', int>();
    program.add_argument("-p", "--print")
            .help("Print results [Boolean]")
            .default_value(false)
            .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(1);
    }
    int nside = program.get<int>("--nside");
    std::string type = program.get<std::string>("--type");
    int L = program.get<int>("--lmax");
    bool print_res = program.get<bool>("--print");

    run_test(nside, type, L, print_res);

    return 0;
}
