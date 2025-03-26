
#include <nanobind/nanobind.h>
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstddef>
#include <complex>
#include <type_traits>

#ifndef NO_CUDA_COMPILER
#include "cuda_runtime.h"
#include "plan_cache.h"
#include "s2fft_kernels.h"
#include "s2fft.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace s2fft {

// =================================================================================================
// Helper template to go from XLA Type to cufft Complex type
// =================================================================================================
template <ffi::DataType DT>
struct FftComplexType;

template <>
struct FftComplexType<ffi::DataType::C128> {
    using type = cufftDoubleComplex;
};

template <>
struct FftComplexType<ffi::DataType::C64> {
    using type = cufftComplex;
};

template <ffi::DataType DT>
using fft_complex_t = typename FftComplexType<DT>::type;

// =================================================================================================
// Helper template to go from XLA Type constexpr boolean indicating if the type is double or not
// =================================================================================================

template <ffi::DataType T>
struct is_double : std::false_type {};

template <>
struct is_double<ffi::DataType::C128> : std::true_type {};

// Helper variable template
template <ffi::DataType T>
constexpr bool is_double_v = is_double<T>::value;

/**
 * @brief Performs the forward spherical harmonic transform.
 *
 * This function executes the forward spherical harmonic transform on the input data
 * using the specified descriptor and CUDA stream.
 *
 * @tparam T The data type of the input and output buffers (e.g., ffi::DataType::C64 or ffi::DataType::C128).
 * @param stream The CUDA stream to associate with the operation.
 * @param input The input buffer containing the data to transform.
 * @param output The output buffer to store the transformed data.
 * @param descriptor The descriptor containing parameters for the transform.
 * @return An ffi::Error indicating the success or failure of the operation.
 */
template <ffi::DataType T>
ffi::Error healpix_forward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                           s2fftDescriptor descriptor) {
    using fft_complex_type = fft_complex_t<T>;
    auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
    fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.untyped_data());
    fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->untyped_data());

    PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
    executor->Forward(descriptor, stream, data_c);
    s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                            stream);

    return ffi::Error::Success();
}

/**
 * @brief Performs the backward spherical harmonic transform.
 *
 * This function executes the backward spherical harmonic transform on the input data
 * using the specified descriptor and CUDA stream.
 *
 * @tparam T The data type of the input and output buffers (e.g., ffi::DataType::C64 or ffi::DataType::C128).
 * @param stream The CUDA stream to associate with the operation.
 * @param input The input buffer containing the data to transform.
 * @param output The output buffer to store the transformed data.
 * @param descriptor The descriptor containing parameters for the transform.
 * @return An ffi::Error indicating the success or failure of the operation.
 */
template <ffi::DataType T>
ffi::Error healpix_backward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                            s2fftDescriptor descriptor) {
    using fft_complex_type = fft_complex_t<T>;

    auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
    fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.untyped_data());
    fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->untyped_data());

    PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
    s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                          descriptor.shift, stream);
    executor->Backward(descriptor, stream, out_c);

    return ffi::Error::Success();
}

/**
 * @brief Constructs a descriptor for the spherical harmonic transform.
 *
 * This function builds a descriptor based on the provided parameters, which is used
 * to configure the spherical harmonic transform operations.
 *
 * @tparam T The data type associated with the descriptor (e.g., ffi::DataType::C64 or ffi::DataType::C128).
 * @param nside The resolution parameter for the transform.
 * @param harmonic_band_limit The maximum harmonic band limit.
 * @param reality Flag indicating if the transform is real-valued.
 * @param forward Flag indicating if the transform is forward (true) or backward (false).
 * @param normalize Flag indicating if the transform should be normalized.
 * @return A s2fftDescriptor configured with the specified parameters.
 */
template <ffi::DataType T>
s2fftDescriptor build_descriptor(int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward,
                                 bool normalize) {
    size_t work_size;
    using fft_complex_type = fft_complex_t<T>;

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

    bool shift = true;

    s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, forward, norm, shift, is_double_v<T>);

    auto executor = std::make_shared<s2fft::s2fftExec<fft_complex_type>>();
    s2fft::PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
    executor->Initialize(descriptor, work_size);

    return descriptor;
}

/**
 * @brief Executes the spherical harmonic transform on the GPU.
 *
 * This function performs the spherical harmonic transform (forward or backward) on the GPU
 * using the specified parameters and CUDA stream.
 *
 * @tparam T The data type of the input and output buffers (e.g., ffi::DataType::C64 or ffi::DataType::C128).
 * @param stream The CUDA stream to associate with the operation.
 * @param nside The resolution parameter for the transform.
 * @param harmonic_band_limit The maximum harmonic band limit.
 * @param reality Flag indicating if the transform is real-value.
 * @param forward Flag indicating if the transform is forward (true) or backward (false).
 * @param normalize Flag indicating if the transform should be normalized.
 * @param input The input buffer containing the data to transform.
 * @param output The output buffer to store the transformed data.
 * @return An ffi::Error indicating the success or failure of the operation.
 */

template <ffi::DataType T>
ffi::Error healpix_fft_cuda(cudaStream_t stream, int64_t nside, int64_t harmonic_band_limit, bool reality,
                            bool forward, bool normalize, ffi::Buffer<T> input,
                            ffi::Result<ffi::Buffer<T>> output) {
    // Get the descriptor from the opaque parameter
    s2fftDescriptor descriptor = build_descriptor<T>(nside, harmonic_band_limit, reality, forward, normalize);
    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.forward) {
        return healpix_forward(stream, input, output, descriptor);
    } else {
        return healpix_backward(stream, input, output, descriptor);
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(healpix_fft_cuda_C64, healpix_fft_cuda<ffi::DataType::C64>,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Attr<int64_t>("nside")
                                      .Attr<int64_t>("harmonic_band_limit")
                                      .Attr<bool>("reality")
                                      .Attr<bool>("forward")
                                      .Attr<bool>("normalize")
                                      .Arg<ffi::Buffer<ffi::DataType::C64>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C64>>()  // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(healpix_fft_cuda_C128, healpix_fft_cuda<ffi::DataType::C128>,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Attr<int64_t>("nside")
                                      .Attr<int64_t>("harmonic_band_limit")
                                      .Attr<bool>("reality")
                                      .Attr<bool>("forward")
                                      .Attr<bool>("normalize")
                                      .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C128>>()  // y
);

template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
    // This check is optional, but it can be helpful for avoiding invalid
    // handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void*>(fn));
}

nb::dict Registration() {
    nb::dict dict;
    dict["healpix_fft_cuda_c64"] = EncapsulateFfiCall(healpix_fft_cuda_C64);
    dict["healpix_fft_cuda_c128"] = EncapsulateFfiCall(healpix_fft_cuda_C128);
    return dict;
}

}  // namespace s2fft

NB_MODULE(_s2fft, m) {
    m.def("registration", &s2fft::Registration);
    m.attr("COMPILED_WITH_CUDA") = true;
}

#else  // NO_CUDA_COMPILER

NB_MODULE(_s2fft, m) {
    m.def("registration", []() { return nb::dict(); });
    m.attr("COMPILED_WITH_CUDA") = false;
}

#endif  // NO_CUDA_COMPILER
