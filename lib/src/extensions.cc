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
 * @param workspace Output buffer for temporary workspace memory.
 * @param callback_params Output buffer for callback parameters.
 * @param descriptor Descriptor containing transform parameters.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_forward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                           ffi::Result<ffi::Buffer<T>> workspace,
                           ffi::Result<ffi::Buffer<ffi::DataType::S64>> callback_params,
                           s2fftDescriptor descriptor) {
    // Step 1: Determine the complex type based on the XLA data type.
    using fft_complex_type = fft_complex_t<T>;
    const auto& dim_in = input.dimensions();

    // Step 2: Handle batched and non-batched cases separately.
    if (dim_in.size() == 2) {
        // Step 2a: Batched case.
        int batch_count = dim_in[0];
        // Step 2b: Compute offsets for input, output, and callback parameters for each batch.
        int64_t input_offset = descriptor.nside * descriptor.nside * 12;
        int64_t output_offset = (4 * descriptor.nside - 1) * (2 * descriptor.harmonic_band_limit);
        int64_t params_offset = 2 * (descriptor.nside - 1) + 1;

        // Step 2c: Fork CUDA streams for parallel processing of batches.
        CudaStreamHandler handler;
        handler.Fork(stream, batch_count);
        auto stream_iter = handler.getIterator();

        // Step 2d: Iterate over each batch.
        for (int i = 0; i < batch_count && stream_iter.hasNext(); ++i) {
            cudaStream_t sub_stream = stream_iter.next();
            // Step 2e: Get or create an s2fftExec instance from the PlanCache.
            auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
            PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);

            // Step 2f: Calculate device pointers for the current batch's data, output, workspace, and
            // callback parameters.
            fft_complex_type* data_c =
                    reinterpret_cast<fft_complex_type*>(input.typed_data() + i * input_offset);
            fft_complex_type* out_c =
                    reinterpret_cast<fft_complex_type*>(output->typed_data() + i * output_offset);
            fft_complex_type* workspace_c =
                    reinterpret_cast<fft_complex_type*>(workspace->typed_data() + i * executor->m_work_size);
            int64* callback_params_c =
                    reinterpret_cast<int64*>(callback_params->typed_data() + i * params_offset);

            // Step 2g: Launch the forward transform on this sub-stream.
            executor->Forward(descriptor, sub_stream, data_c, workspace_c, callback_params_c);
            // Step 2h: Launch spectral extension kernel.
            s2fftKernels::launch_spectral_extension(data_c, out_c, descriptor.nside,
                                                    descriptor.harmonic_band_limit, sub_stream);
        }
        // Step 2i: Join all forked streams back to the main stream.
        handler.join(stream);
        return ffi::Error::Success();
    } else {
        // Step 2j: Non-batched case.
        // Step 2k: Get device pointers for data, output, workspace, and callback parameters.
        fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.typed_data());
        fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->typed_data());
        fft_complex_type* workspace_c = reinterpret_cast<fft_complex_type*>(workspace->typed_data());
        int64* callback_params_c = reinterpret_cast<int64*>(callback_params->typed_data());

        // Step 2l: Get or create an s2fftExec instance from the PlanCache.
        auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Step 2m: Launch the forward transform.
        executor->Forward(descriptor, stream, data_c, workspace_c, callback_params_c);
        // Step 2n: Launch spectral extension kernel.
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
 * @param workspace Output buffer for temporary workspace memory.
 * @param callback_params Output buffer for callback parameters.
 * @param descriptor Descriptor containing transform parameters.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_backward(cudaStream_t stream, ffi::Buffer<T> input, ffi::Result<ffi::Buffer<T>> output,
                            ffi::Result<ffi::Buffer<T>> workspace,
                            ffi::Result<ffi::Buffer<ffi::DataType::S64>> callback_params,
                            s2fftDescriptor descriptor) {
    // Step 1: Determine the complex type based on the XLA data type.
    using fft_complex_type = fft_complex_t<T>;
    const auto& dim_in = input.dimensions();
    const auto& dim_out = output->dimensions();

    // Step 2: Handle batched and non-batched cases separately.
    if (dim_in.size() == 3) {
        // Step 2a: Batched case.
        // Assertions to ensure correct input/output dimensions for batched operations.
        assert(dim_out.size() == 2);
        assert(dim_in[0] == dim_out[0]);
        int batch_count = dim_in[0];
        // Step 2b: Compute offsets for input, output, and callback parameters for each batch.
        int64_t input_offset = (4 * descriptor.nside - 1) * (2 * descriptor.harmonic_band_limit);
        int64_t output_offset = descriptor.nside * descriptor.nside * 12;

        // Step 2c: Fork CUDA streams for parallel processing of batches.
        CudaStreamHandler handler;
        handler.Fork(stream, batch_count);
        auto stream_iter = handler.getIterator();

        // Step 2d: Iterate over each batch.
        for (int i = 0; i < batch_count && stream_iter.hasNext(); ++i) {
            cudaStream_t sub_stream = stream_iter.next();
            // Step 2e: Get or create an s2fftExec instance from the PlanCache.
            auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
            PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);

            // Step 2f: Calculate device pointers for the current batch's data, output, workspace, and
            // callback parameters.
            fft_complex_type* data_c =
                    reinterpret_cast<fft_complex_type*>(input.typed_data() + i * input_offset);
            fft_complex_type* out_c =
                    reinterpret_cast<fft_complex_type*>(output->typed_data() + i * output_offset);
            fft_complex_type* workspace_c =
                    reinterpret_cast<fft_complex_type*>(workspace->typed_data() + i * executor->m_work_size);
            int64* callback_params_c =
                    reinterpret_cast<int64*>(callback_params->typed_data() + i * sizeof(int64) * 2);

            // Step 2g: Launch spectral folding kernel.
            s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside,
                                                  descriptor.harmonic_band_limit, descriptor.shift,
                                                  sub_stream);
            // Step 2h: Launch the backward transform on this sub-stream.
            executor->Backward(descriptor, sub_stream, out_c, workspace_c, callback_params_c);
        }
        // Step 2i: Join all forked streams back to the main stream.
        handler.join(stream);
        return ffi::Error::Success();
    } else {
        // Step 2j: Non-batched case.
        // Assertions to ensure correct input/output dimensions for non-batched operations.
        assert(dim_in.size() == 2);
        assert(dim_out.size() == 1);
        // Step 2k: Get device pointers for data, output, workspace, and callback parameters.
        fft_complex_type* data_c = reinterpret_cast<fft_complex_type*>(input.typed_data());
        fft_complex_type* out_c = reinterpret_cast<fft_complex_type*>(output->typed_data());
        fft_complex_type* workspace_c = reinterpret_cast<fft_complex_type*>(workspace->typed_data());
        int64* callback_params_c = reinterpret_cast<int64*>(callback_params->typed_data());

        // Step 2l: Get or create an s2fftExec instance from the PlanCache.
        auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Step 2m: Launch spectral folding kernel.
        s2fftKernels::launch_spectral_folding(data_c, out_c, descriptor.nside, descriptor.harmonic_band_limit,
                                              descriptor.shift, stream);
        // Step 2n: Launch the backward transform.
        executor->Backward(descriptor, stream, out_c, workspace_c, callback_params_c);
        return ffi::Error::Success();
    }
}

/**
 * @brief Builds an s2fftDescriptor based on provided parameters.
 *
 * This descriptor is identical for all batch elements. It also ensures that
 * an s2fftExec instance corresponding to the descriptor is initialized and cached.
 *
 * @tparam T The XLA data type.
 * @param nside HEALPix resolution parameter.
 * @param harmonic_band_limit Harmonic band limit L.
 * @param reality Flag indicating whether data is real-valued.
 * @param forward Flag indicating forward transform.
 * @param normalize Flag for normalization.
 * @param adjoint Flag indicating if an adjoint operation is desired.
 * @param must_exist If true, throws an error if the plan does not exist in the cache.
 * @return s2fftDescriptor configured with the given parameters.
 */
template <ffi::DataType T>
s2fftDescriptor build_descriptor(int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward,
                                 bool normalize, bool adjoint, bool must_exist , size_t& work_size) {
    using fft_complex_type = fft_complex_t<T>;
    // Step 1: Determine FFT normalization type based on forward/normalize flags.
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
    // Step 2: Set shift flag (always true for now).
    bool shift = true;
    // Step 3: Create an s2fftDescriptor object with the given parameters.
    s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, adjoint, forward, norm, shift,
                               is_double_v<T>);

    // Step 4: Get or create an s2fftExec instance from the PlanCache.
    // This call will also initialize the executor if it's newly created.
    auto executor = std::make_shared<s2fftExec<fft_complex_type>>();
    HRESULT hr = PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
    // Step 5: Handle cases where the plan was expected to exist but didn't.
    if (hr == S_OK && must_exist) {
        // This is an error because S_OK means plan was created, but must_exist implies it should have been
        // found.
        throw std::runtime_error("S2FFT INTERNAL ERROR: Plan did not exist but it was expected to exist.");
    }
    // Step 6: If the executor was just created (S_OK), initialize it.
    // Note: PlanCache::GetS2FFTExec now handles workspace initialization internally
    if (hr == S_OK) {
        executor->Initialize(descriptor);
    }
    // Make sure workspace is set 
    assert(executor->m_work_size > 0 && "S2FFT INTERNAL ERROR: Workspace size is zero after initialization.");
    work_size = executor->m_work_size;
    // Step 7: Return the created descriptor.
    return descriptor;
}

/**
 * @brief Unified entry point for the HEALPix FFT transform.
 *
 * This function serves as the main FFI entry point for HEALPix FFT operations.
 * Depending on the value of the 'forward' flag in the descriptor, it dispatches
 * to either the forward (`healpix_forward`) or backward (`healpix_backward`) transform.
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
 * @param workspace Output buffer for temporary workspace memory.
 * @param callback_params Output buffer for callback parameters.
 * @return ffi::Error indicating success or failure.
 */
template <ffi::DataType T>
ffi::Error healpix_fft_cuda(cudaStream_t stream, int64_t nside, int64_t harmonic_band_limit, bool reality,
                            bool forward, bool normalize, bool adjoint, ffi::Buffer<T> input,
                            ffi::Result<ffi::Buffer<T>> output, ffi::Result<ffi::Buffer<T>> workspace,
                            ffi::Result<ffi::Buffer<ffi::DataType::S64>> callback_params) {
    // Step 1: Build the s2fftDescriptor based on the input parameters.
    size_t work_size = 0;  // Variable to hold the workspace size
    s2fftDescriptor descriptor =
            build_descriptor<T>(nside, harmonic_band_limit, reality, forward, normalize, adjoint, true , work_size);

    // Step 2: Dispatch to either forward or backward transform based on the 'forward' flag.
    if (forward) {
        return healpix_forward<T>(stream, input, output, workspace, callback_params, descriptor);
    } else {
        return healpix_backward<T>(stream, input, output, workspace, callback_params, descriptor);
    }
}

/**
 * @brief FFI registration for the HEALPix FFT CUDA functions.
 *
 * Registers the handlers for both C64 and C128 data types with XLA FFI.
 * This makes the CUDA FFT functions callable from JAX.
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
                                      .Ret<ffi::Buffer<ffi::DataType::C64>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C64>>()
                                      .Ret<ffi::Buffer<ffi::DataType::S64>>());

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
                                      .Ret<ffi::Buffer<ffi::DataType::C128>>()
                                      .Ret<ffi::Buffer<ffi::DataType::C128>>()
                                      .Ret<ffi::Buffer<ffi::DataType::S64>>());

/**
 * @brief Encapsulates an FFI handler into a nanobind capsule.
 *
 * This helper function is used to wrap C++ FFI handlers so they can be exposed
 * to Python via nanobind.
 *
 * @tparam T The function type of the FFI handler.
 * @param fn Pointer to the FFI handler function.
 * @return nb::capsule A nanobind capsule containing the FFI handler.
 */
template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
    // Step 1: Assert that the provided function is a valid XLA FFI handler.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "Encapsulated function must be an XLA FFI handler");
    // Step 2: Return a nanobind capsule wrapping the function pointer.
    return nb::capsule(reinterpret_cast<void*>(fn));
}

/**
 * @brief Returns a dictionary of all registered FFI handlers.
 *
 * This function creates a nanobind dictionary that maps string names to
 * encapsulated FFI handlers, allowing them to be looked up and called from Python.
 *
 * @return nb::dict A nanobind dictionary with keys for each handler.
 */
nb::dict Registration() {
    // Step 1: Create an empty nanobind dictionary.
    nb::dict dict;
    // Step 2: Add encapsulated FFI handlers for C64 and C128 to the dictionary.
    dict["healpix_fft_cuda_c64"] = EncapsulateFfiCall(healpix_fft_cuda_C64);
    dict["healpix_fft_cuda_c128"] = EncapsulateFfiCall(healpix_fft_cuda_C128);
    // Step 3: Return the populated dictionary.
    return dict;
}

}  // namespace s2fft

NB_MODULE(_s2fft, m) {
    // Step 1: Expose the registration function to Python.
    m.def("registration", &s2fft::Registration);
    // Step 2: Declare and expose build_descriptor functions for C64 and C128 to Python.
    // These functions allow Python to query the required workspace size for a given descriptor.
    m.def("build_descriptor_C64", [](int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward,
                                     bool normalize, bool adjoint) {
        // Step 2a: Build the s2fftDescriptor.
        size_t work_size = 0;  // Variable to hold the workspace size
        s2fft::s2fftDescriptor desc = s2fft::build_descriptor<ffi::DataType::C64>(
                nside, harmonic_band_limit, reality, forward, normalize, adjoint, false, work_size);
        return work_size;
    });
    m.def("build_descriptor_C128", [](int64_t nside, int64_t harmonic_band_limit, bool reality, bool forward,
                                      bool normalize, bool adjoint) {
        // Step 2e: Build the s2fftDescriptor.
        size_t work_size = 0;  // Variable to hold the workspace size
        s2fft::s2fftDescriptor desc = s2fft::build_descriptor<ffi::DataType::C128>(
                nside, harmonic_band_limit, reality, forward, normalize, adjoint, false, work_size);
        return work_size;
    });
    // Step 3: Expose a boolean attribute indicating if CUDA support is compiled in.
    m.attr("COMPILED_WITH_CUDA") = true;
}

#else  // NO_CUDA_COMPILER

// Step 1: Define a fallback NB_MODULE when CUDA is not compiled.
NB_MODULE(_s2fft, m) {
    // Step 1a: Provide a dummy registration function that returns an empty dictionary.
    m.def("registration", []() { return nb::dict(); });
    // Step 1b: Indicate that CUDA support is not compiled.
    m.attr("COMPILED_WITH_CUDA") = false;
}

#endif  // NO_CUDA_COMPILER