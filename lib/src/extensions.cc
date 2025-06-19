#include <nanobind/nanobind.h>
#include <cstddef>
#include <complex>
#include <type_traits>

namespace nb = nanobind;

#ifndef NO_CUDA_COMPILER
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "cuda_runtime.h"
#include "plan_cache.h"
#include "s2fft_kernels.h"
#include "s2fft.h"
#include "cudastreamhandler.hpp"  // For forking and joining CUDA streams

namespace ffi = xla::ffi;

namespace s2fft {

/**
 * @brief Mapping from XLA DataType to CUFFT complex types.
 */
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

/**
 * @brief Helper to indicate if using double precision.
 *
 * Default is false; specialized for C128.
 */
template <ffi::DataType T>
struct is_double : std::false_type {};

template <>
struct is_double<ffi::DataType::C128> : std::true_type {};

template <ffi::DataType T>
constexpr bool is_double_v = is_double<T>::value;

/**
 * @brief Performs a forward HEALPix transform on a single element or batch.
 *
 * For a batched call, the input buffer is assumed to be 2D: [batch_size, nside^2*12],
 * and the output is 3D: [batch_size, (4*nside-1), 2*harmonic_band_limit].
 *
 * For non-batched call, the input is 1D and the output is 1D.
 *
 * @tparam T The XLA data type (F32, F64, etc).
 * @param stream CUDA stream to use.
 * @param input Input buffer containing HEALPix pixel-space data.
 * @param output Output buffer to store the FTM result.
 * @param descriptor Descriptor containing transform parameters.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_forward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                           s2fftDescriptor descriptor) {
    using fft_complex_type = fft_complex_t<T>;
    const auto& dim_in = input.dimensions();

    if (dim_in.size() == 2) {
        // Batched case.
        int batch_count = dim_in[0];
        // Compute per-batch offset (number of elements per batch).
        int64_t input_offset = descriptor.nside * descriptor.nside * 12;
        int64_t output_offset = (4 * descriptor.nside - 1) * (2 * descriptor.harmonic_band_limit);

        CudaStreamHandler handler;
        handler.Fork(stream, batch_count);
        auto stream_iter = handler.getIterator();

        for (int i = 0; i < batch_count && stream_iter.hasNext(); ++i) {
            cudaStream_t sub_stream = stream_iter.next();
            fft_complex_type* data_c =
                    reinterpret_cast<fft_complex_type*>(input.typed_data() + i * input_offset);
            fft_complex_type* out_c =
                    reinterpret_cast<fft_complex_type*>(output->typed_data() + i * output_offset);

            auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
            PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
            // Launch the forward transform on this sub-stream.
            executor->Forward(descriptor, sub_stream, data_c);
            s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside,
                                                    descriptor.harmonic_band_limit, sub_stream);
        }
        handler.join(stream);
        return ffi::Error::Success();
    } else {
        // Non-batched case.
        fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.typed_data());
        fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->typed_data());

        auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        executor->Forward(descriptor, stream, data_c);
        s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside,
                                                descriptor.harmonic_band_limit, stream);
        return ffi::Error::Success();
    }
}

/**
 * @brief Performs a backward HEALPix transform on a single element or batch.
 *
 * For a batched call, the input buffer is assumed to be 3D: [batch_size, (4*nside-1), 2*harmonic_band_limit],
 * and the output is 2D: [batch_size, nside^2*12].
 *
 * For non-batched call, the input is 1D and the output is 1D.
 *
 * @tparam T The XLA data type.
 * @param stream CUDA stream to use.
 * @param input Input buffer containing FTM data.
 * @param output Output buffer to store HEALPix pixel-space data.
 * @param descriptor Descriptor containing transform parameters.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_backward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                            s2fftDescriptor descriptor) {
    using fft_complex_type = fft_complex_t<T>;
    const auto& dim_in = input.dimensions();
    const auto& dim_out = output->dimensions();

    if (dim_in.size() == 3) {
        // Batched case.
        assert(dim_out.size() == 2);
        assert(dim_in[0] == dim_out[0]);
        int batch_count = dim_in[0];
        int64_t input_offset = (4 * descriptor.nside - 1) * (2 * descriptor.harmonic_band_limit);
        int64_t output_offset = descriptor.nside * descriptor.nside * 12;

        CudaStreamHandler handler;
        handler.Fork(stream, batch_count);
        auto stream_iter = handler.getIterator();

        for (int i = 0; i < batch_count && stream_iter.hasNext(); ++i) {
            cudaStream_t sub_stream = stream_iter.next();
            fft_complex_type* data_c =
                    reinterpret_cast<fft_complex_type*>(input.typed_data() + i * input_offset);
            fft_complex_type* out_c =
                    reinterpret_cast<fft_complex_type*>(output->typed_data() + i * output_offset);

            auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
            PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
            s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside,
                                                  descriptor.harmonic_band_limit, descriptor.shift,
                                                  sub_stream);
            executor->Backward(descriptor, sub_stream, out_c);
        }
        handler.join(stream);
        return ffi::Error::Success();
    } else {
        // Non-batched case.
        assert(dim_in.size() == 2);
        assert(dim_out.size() == 1);
        fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.typed_data());
        fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->typed_data());

        auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                              descriptor.shift, stream);
        executor->Backward(descriptor, stream, out_c);
        return ffi::Error::Success();
    }
}

/**
 * @brief Builds an s2fftDescriptor based on provided parameters.
 *
 * This descriptor is identical for all batch elements.
 *
 * @tparam T The XLA data type.
 * @param nside HEALPix resolution parameter.
 * @param harmonic_band_limit Harmonic band limit L.
 * @param reality Flag indicating whether data is real-valued.
 * @param forward Flag indicating forward transform.
 * @param normalize Flag for normalization.
 * @param adjoint Flag indicating if an adjoint operation is desired.
 * @return s2fftDescriptor configured with the given parameters.
 */
template <ffi::DataType T>
s2fftDescriptor build_descriptor(int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward,
                                 bool normalize, bool adjoint) {
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
    s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, adjoint, forward, norm, shift,
                               is_double_v<T>);
    auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
    PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
    executor->Initialize(descriptor, work_size);
    return descriptor;
}

/**
 * @brief Unified entry point for the HEALPix FFT transform.
 *
 * Depending on the value of the 'forward' flag, it dispatches to either the forward or backward transform.
 *
 * @tparam T The XLA data type.
 * @param stream CUDA stream to use.
 * @param nside HEALPix resolution parameter.
 * @param harmonic_band_limit Harmonic band limit L.
 * @param reality Flag indicating whether data is real-valued.
 * @param forward Flag indicating forward transform.
 * @param normalize Flag for normalization.
 * @param adjoint Flag indicating if an adjoint operation is desired.
 * @param input Input buffer.
 * @param output Output buffer.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_fft_cuda(cudaStream_t stream, int64_t nside, int64_t harmonic_band_limit, bool reality,
                            bool forward, bool normalize, bool adjoint, ffi::Buffer<T> input,
                            ffi::Result<ffi::Buffer<T>> output) {
    s2fftDescriptor descriptor =
            build_descriptor<T>(nside, harmonic_band_limit, reality, forward, normalize, adjoint);

    if (forward) {
        return healpix_forward<T>(stream, input, output, descriptor);
    } else {
        return healpix_backward<T>(stream, input, output, descriptor);
    }
}

/**
 * @brief FFI registration for the HEALPix FFT CUDA functions.
 *
 * Registers the handlers for both C64 and C128 data types.
 */
XLA_FFI_DEFINE_HANDLER_SYMBOL(healpix_fft_cuda_C64, healpix_fft_cuda<ffi::DataType::C64>,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Attr<int64_t>("nside")
                                      .Attr<int64_t>("harmonic_band_limit")
                                      .Attr<bool>("reality")
                                      .Attr<bool>("forward")
                                      .Attr<bool>("normalize")
                                      .Attr<bool>("adjoint")
                                      .Arg<ffi::Buffer<ffi::DataType::C64>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(healpix_fft_cuda_C128, healpix_fft_cuda<ffi::DataType::C128>,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Attr<int64_t>("nside")
                                      .Attr<int64_t>("harmonic_band_limit")
                                      .Attr<bool>("reality")
                                      .Attr<bool>("forward")
                                      .Attr<bool>("normalize")
                                      .Attr<bool>("adjoint")
                                      .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C128>>());

/**
 * @brief Encapsulates an FFI handler into a nanobind capsule.
 *
 * @tparam T The function type.
 * @param fn Pointer to the FFI handler.
 * @return nb::capsule encapsulating the handler.
 */
template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "Encapsulated function must be an XLA FFI handler");
    return nb::capsule(reinterpret_cast<void*>(fn));
}

/**
 * @brief Returns a dictionary of all registered FFI handlers.
 *
 * @return nb::dict with keys for each handler.
 */
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
