
#include "kernel_nanobind_helpers.h"
#include "kernel_helpers.h"
#include <nanobind/nanobind.h>
#include <cstddef>
#include "cuda_runtime.h"
#include "plan_cache.h"
#include "s2fft_kernels.h"
#include "s2fft.h"

namespace nb = nanobind;

namespace s2fft {

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
        s2fftKernels::launch_spectral_extension<cufftDoubleComplex>(data_c, out_c, descriptor.nside,
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
        s2fftKernels::launch_spectral_folding<cufftDoubleComplex>(data_c, out_c, descriptor.nside,
                                                                  descriptor.harmonic_band_limit, stream);
        // Run the fft part
        executor->Backward(descriptor, stream, out_c);

    } else {
        auto executor = std::make_shared<s2fftExec<cufftComplex>>();
        cufftComplex* data_c = reinterpret_cast<cufftComplex*>(data);
        cufftComplex* out_c = reinterpret_cast<cufftComplex*>(output);

        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the spectral folding part
        s2fftKernels::launch_spectral_folding<cufftComplex>(data_c, out_c, descriptor.nside,
                                                            descriptor.harmonic_band_limit, stream);
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

std::pair<int64_t, nb::bytes> build_healpix_fft_descriptor(int nside, int harmonic_band_limit, bool reality,
                                                           bool forward, bool double_precision) {
    size_t work_size;
    // Only backward for now
    s2fftKernels::fft_norm norm = s2fftKernels::fft_norm::BACKWARD;
    // Always shift
    bool shift = true;
    s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, forward, norm, shift, double_precision);

    if (double_precision) {
        auto executor = std::make_shared<s2fftExec<cufftDoubleComplex>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        executor->Initialize(descriptor, work_size);
        return std::pair<int64_t, nb::bytes>(work_size, PackDescriptor(descriptor));
    } else {
        auto executor = std::make_shared<s2fftExec<cufftComplex>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        executor->Initialize(descriptor, work_size);
        return std::pair<int64_t, nb::bytes>(work_size, PackDescriptor(descriptor));
    }
}

nb::dict Registration() {
    nb::dict dict;
    dict["healpix_fft_cuda"] = EncapsulateFunction(healpix_fft_cuda);
    return dict;
}

}  // namespace s2fft

NB_MODULE(_s2fft, m) {
    m.def("registration", &s2fft::Registration);

    m.def("build_healpix_fft_descriptor",
          [](int nside, int harmonic_band_limit, bool reality, bool forward, bool double_precision) {
              size_t work_size;
              // Only backward for now
              s2fftKernels::fft_norm norm = s2fftKernels::fft_norm::BACKWARD;
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
          });
}