#include <algorithm>
#include "s2fft.h"
#include "s2fft_kernels.h"
#include <array>
#include <iostream>
#include <ostream>
#include <sstream>
#include "cufft.h"
#include "s2fft_kernels.h"
#include <argparse.hpp>

using namespace s2fft;
using namespace s2fftKernels;

enum FFTType { FORWARD, BACKWARD, BOTH, NONE };

using complex = cufftComplex;

void run_test(int nside, FFTType ffttype, bool shift, fft_norm norm, bool test, bool print_res) {
    int L = 2 * nside;
    int total_pixels = 12 * nside * nside;

    // Compute the flm size
    int polar_pixels = 4 * nside * (nside - 1);
    int equator_rings_num = (total_pixels - polar_pixels) / (4 * nside);
    int num_rings = equator_rings_num + 2 * (nside - 1);
    int flm_size = num_rings * (4 * nside);

    int input_size = ffttype == FFTType::FORWARD ? total_pixels : flm_size;
    int output_size = ffttype == FFTType::FORWARD ? flm_size : total_pixels;
    // input_size = output_size;

    complex *h_vec_in = new complex[input_size];
    complex *h_vec_out = new complex[output_size];
    complex *d_vec_in;
    complex *d_vec_out;

    cudaMalloc(&d_vec_in, input_size * sizeof(complex));
    cudaMalloc(&d_vec_out, output_size * sizeof(complex));

    // Initialize host vectors using std::generate
    int start_index(0);
    std::generate(h_vec_in, h_vec_in + input_size, [&start_index]() {
        complex c;
        c.x = start_index;
        c.y = 0.0f;
        start_index += 1;
        return c;
    });

    // Copy host data to device
    cudaMemcpy(d_vec_in, h_vec_in, input_size * sizeof(complex), cudaMemcpyHostToDevice);

    ////// Create cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    s2fftDescriptor desc(nside, L, true, true, norm, shift);
    s2fftExec<complex> exec;
    size_t worksize(0);
    exec.Initialize(desc, worksize);

    void **buffers = (void **)malloc(sizeof(void *));
    buffers[0] = d_vec_in;

    ////// ********************************************************
    ////// Perform forward
    ////// ********************************************************
    if (ffttype == FFTType::FORWARD || ffttype == FFTType::BOTH) {
        exec.Forward(desc, stream, d_vec_in);
        s2fftKernels::launch_spectral_extension(d_vec_in, d_vec_out, nside, L, stream);
    }
    if (ffttype == FFTType::BACKWARD || ffttype == FFTType::BOTH) {
        s2fftKernels::launch_spectral_folding(d_vec_in, d_vec_out, nside, L, stream);

        // exec.Backward(desc, stream, d_vec_out);
        cudaStreamSynchronize(stream);
    }
    // Spectral extension

    cudaStreamSynchronize(stream);
    //// Copy device data to host
    cudaMemcpy(h_vec_out, d_vec_out, output_size * sizeof(complex), cudaMemcpyDeviceToHost);

    std::cout << "After FFT" << std::endl;

    if (print_res)
        for (int i = 0; i < output_size; i++) {
            std::cout << "[" << i << "] " << h_vec_out[i].x << " + " << h_vec_out[i].y << "i" << std::endl;
        }

    // test
    if (test) {
        float max_error = 0.0f;
        for (int i = 0; i < input_size; i++) {
            float error = std::max(std::abs(h_vec_in[i].x - h_vec_out[i].x),
                                   std::abs(h_vec_in[i].y - h_vec_out[i].y));
            if (error > 0.5f) {
                std::cout << "Element: " << i << " Error: " << error << std::endl;
            }
            max_error = std::max(max_error, error);
        }
        std::cout << "Max error: " << max_error << std::endl;
    }

    // Free memory
    delete[] h_vec_in;
    delete[] h_vec_out;
    cudaFree(d_vec_in);
    cudaFree(d_vec_out);
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("s2fft_test");
    program.add_argument("-s", "--nside").help("nside").scan<'i', int>();
    program.add_argument("-t", "--ffttype").help("FFTType : fwd bwd both").default_value("both");
    program.add_argument("-sh", "--shift")
            .help("fftshift result [Boolean]")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("-n", "--norm").help("Normalization : bwd fwd ortho").default_value("bwd");
    program.add_argument("-c", "--check")
            .help("Check result [Boolean]")
            .default_value(false)
            .implicit_value(true);
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
    std::string ffttype = program.get<std::string>("--ffttype");
    bool shift = program.get<bool>("--shift");
    std::string norm = program.get<std::string>("--norm");
    bool test = program.get<bool>("--check");
    bool print_res = program.get<bool>("--print");

    FFTType type = FFTType::NONE;

    if (ffttype == "fwd") {
        type = FFTType::FORWARD;
    } else if (ffttype == "bwd") {
        type = FFTType::BACKWARD;
    } else {
        type = FFTType::BOTH;
    }

    fft_norm norm_type = fft_norm::NONE;
    if (norm == "ortho") {
        norm_type = fft_norm::ORTHO;
    } else if (norm == "fwd") {
        norm_type = fft_norm::FORWARD;
    } else {
        norm_type = fft_norm::BACKWARD;
    }

    run_test(nside, type, shift, norm_type, test, print_res);

    return 0;
}
