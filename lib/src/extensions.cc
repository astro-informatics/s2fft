
#include "kernel_nanobind_helpers.h"
#include "kernel_helpers.h"
#include <nanobind/nanobind.h>
#include <cstddef>

#ifndef NO_CUDA_COMPILER
#include "cuda_runtime.h"
#include "plan_cache.h"
#include "s2fft_kernels.h"
#include "s2fft.h"
#else
void print_error() {
  
    throw std::runtime_error("This extension was compiled without CUDA support. Cuda functions are not supported.");
}
#endif

namespace nb = nanobind;

namespace s2fft {

#ifdef NO_CUDA_COMPILER
void healpix_fft_cuda() { print_error(); }
#else
void healpix_forward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
    void* data = buffers[0];
    void* output = buffers[1];

    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.double_precision) {
        auto executor = std::make_shared<s2fftExec<cufftDoubleComplex>>();
        cufftDoubleComplex* data_c = reinterpret_cast<cufftDoubleComplex*>(data);
        cufftDoubleComplex* out_c = reinterpret_cast<cufftDoubleComplex*>(output);

        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the fft part
        executor->Forward(descriptor, stream, data_c);
        // Run the spectral extension part
        s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside,
                                                descriptor.harmonic_band_limit, stream);

    } else {
        auto executor = std::make_shared<s2fftExec<cufftComplex>>();
        cufftComplex* data_c = reinterpret_cast<cufftComplex*>(data);
        cufftComplex* out_c = reinterpret_cast<cufftComplex*>(output);

        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the fft part
        executor->Forward(descriptor, stream, data_c);
        // Run the spectral extension part
        s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside,
                                                descriptor.harmonic_band_limit, stream);
    }
}

void healpix_backward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
    void* data = buffers[0];
    void* output = buffers[1];

    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.double_precision) {
        auto executor = std::make_shared<s2fftExec<cufftDoubleComplex>>();
        cufftDoubleComplex* data_c = reinterpret_cast<cufftDoubleComplex*>(data);
        cufftDoubleComplex* out_c = reinterpret_cast<cufftDoubleComplex*>(output);

        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the spectral folding part
        s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                              descriptor.shift, stream);
        // Run the fft part
        executor->Backward(descriptor, stream, out_c);

    } else {
        auto executor = std::make_shared<s2fftExec<cufftComplex>>();
        cufftComplex* data_c = reinterpret_cast<cufftComplex*>(data);
        cufftComplex* out_c = reinterpret_cast<cufftComplex*>(output);

        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the spectral folding part
        s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                              descriptor.shift, stream);
        // Run the fft part
        executor->Backward(descriptor, stream, out_c);
    }
}

void healpix_fft_cuda(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    // Get the descriptor from the opaque parameter
    s2fftDescriptor descriptor = *UnpackDescriptor<s2fftDescriptor>(opaque, opaque_len);
    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.forward) {
        healpix_forward(stream, buffers, descriptor);
    } else {
        healpix_backward(stream, buffers, descriptor);
    }
}

#endif  // NO_CUDA_COMPILER

nb::dict Registration() {
    nb::dict dict;
    dict["healpix_fft_cuda"] = EncapsulateFunction(healpix_fft_cuda);
    return dict;
}

}  // namespace s2fft

NB_MODULE(_s2fft, m) {
    m.def("registration", &s2fft::Registration);

    m.def("build_healpix_fft_descriptor",
          [](int nside, int harmonic_band_limit, bool reality, bool forward,bool normalize, bool double_precision) {
#ifndef NO_CUDA_COMPILER
              size_t work_size;
              // Only backward for now
              s2fftKernels::fft_norm norm = s2fftKernels::fft_norm::NONE;
              if (forward && normalize) {
                  norm = s2fftKernels::fft_norm::FORWARD;
              } else if (!forward && normalize) {
                  norm = s2fftKernels::fft_norm::BACKWARD;
              } else if (forward && !normalize) {
                  norm = s2fftKernels::fft_norm::BACKWARD;
              } else if (!forward && !normalize) {
                  norm = s2fftKernels::fft_norm::FORWARD;
              }
              // Always shift
              bool shift = true;
              s2fft::s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, forward, norm, shift,
                                                double_precision);

              if (double_precision) {
                  auto executor = std::make_shared<s2fft::s2fftExec<cufftDoubleComplex>>();
                  s2fft::PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
                  executor->Initialize(descriptor, work_size);
                  return PackDescriptor(descriptor);
              } else {
                  auto executor = std::make_shared<s2fft::s2fftExec<cufftComplex>>();
                  s2fft::PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
                  executor->Initialize(descriptor, work_size);
                  return PackDescriptor(descriptor);
              }
#else
              print_error();
#endif
          });
}
