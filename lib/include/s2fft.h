#ifndef S2FFT_H
#define S2FFT_H

#define FORWARD_NORM 1
#define BACKWARD_NORM 2
#define ORTHO_NORM 3
#define NONE_NORM 4

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cuda/std/complex>
#include <iostream>
#include <vector>
#include "cufft.h"
#include "cufftXt.h"
#include "thrust/device_vector.h"
#include "s2fft_kernels.h"


namespace s2fft {

/**
 * @brief Returns the appropriate cuFFT C2C type for a given complex type.
 *
 * This function is overloaded for `cufftDoubleComplex` and `cufftComplex`
 * to return `CUFFT_Z2Z` (double precision) or `CUFFT_C2C` (single precision)
 * respectively.
 *
 * @param dummy A dummy complex object used for type deduction.
 * @return The corresponding cuFFT C2C type.
 */
static cufftType get_cufft_type_c2c(cufftDoubleComplex) { return CUFFT_Z2Z; }
static cufftType get_cufft_type_c2c(cufftComplex) { return CUFFT_C2C; }

/**
 * @brief Transforms data from ring-based indexing to nphi-based indexing.
 *
 * This function is a placeholder for the actual implementation which would
 * reorder data in memory according to the specified indexing scheme.
 *
 * @param data Pointer to the input/output data.
 * @param nside The HEALPix Nside parameter.
 */
void s2fft_rings_2_nphi(float *data, int nside);

/**
 * @brief Transforms data from nphi-based indexing to ring-based indexing.
 *
 * This function is a placeholder for the actual implementation which would
 * reorder data in memory according to the specified indexing scheme.
 *
 * @param data Pointer to the input/output data.
 * @param nside The HEALPix Nside parameter.
 */
void s2fft_nphi_2_rings(float *data, int nside);

/**
 * @brief Descriptor class for s2fft operations.
 *
 * This class encapsulates all the necessary parameters to define a unique
 * Spherical Harmonic Transform (SHT) operation, including Nside, harmonic
 * band limit, reality, adjoint flag, forward/backward transform direction,
 * normalization, shifting, and double precision usage.
 */
class s2fftDescriptor {
public:
    int64_t nside;
    int64_t harmonic_band_limit;
    bool reality;
    bool adjoint;

    bool forward = true;
    s2fftKernels::fft_norm norm = s2fftKernels::BACKWARD;
    bool shift = true;
    bool double_precision = false;

    /**
     * @brief Constructs an s2fftDescriptor object.
     *
     * @param nside The HEALPix Nside parameter.
     * @param harmonic_band_limit The harmonic band limit L.
     * @param reality Flag indicating if the signal is real.
     * @param adjoint Flag indicating if the adjoint transform is to be performed.
     * @param forward Flag indicating if it's a forward transform (default: true).
     * @param norm The FFT normalization type (default: BACKWARD).
     * @param shift Flag indicating if FFT shifting should be applied (default: true).
     * @param double_precision Flag indicating if double precision should be used (default: false).
     */
    s2fftDescriptor(int64_t nside, int64_t harmonic_band_limit, bool reality, bool adjoint,
                    bool forward = true, s2fftKernels::fft_norm norm = s2fftKernels::BACKWARD,
                    bool shift = true, bool double_precision = false)
            : nside(nside),
              harmonic_band_limit(harmonic_band_limit),
              reality(reality),
              adjoint(adjoint),
              norm(norm),
              forward(forward),
              shift(shift),
              double_precision(double_precision) {}

    /**
     * @brief Default constructor for s2fftDescriptor.
     */
    s2fftDescriptor() = default;

    /**
     * @brief Destructor for s2fftDescriptor.
     */
    ~s2fftDescriptor() = default;

    /**
     * @brief Equality operator for s2fftDescriptor.
     *
     * Compares two s2fftDescriptor objects for equality based on their member values.
     *
     * @param other The other s2fftDescriptor to compare against.
     * @return True if the descriptors are equal, false otherwise.
     */
    bool operator==(const s2fftDescriptor &other) const {
        return nside == other.nside && harmonic_band_limit == other.harmonic_band_limit &&
               reality == other.reality && norm == other.norm && shift == other.shift &&
               double_precision == other.double_precision;
    }
};

/**
 * @brief Executes Spherical Harmonic Transform (SHT) operations.
 *
 * This templated class provides methods for initializing FFT plans and executing
 * forward and backward SHTs. It manages cuFFT handles and internal offsets
 * required for the transforms.
 *
 * @tparam Complex The complex type (cufftComplex or cufftDoubleComplex) for the FFT operations.
 */
template <typename Complex>
class s2fftExec {
    friend class PlanCache;  // Allows PlanCache to access private members for caching

public:
    /**
     * @brief Default constructor for s2fftExec.
     */
    s2fftExec() {}

    /**
     * @brief Destructor for s2fftExec.
     */
    ~s2fftExec() {}

    /**
     * @brief Initializes the FFT plans for the SHT.
     *
     * This method sets up the necessary cuFFT plans for both polar and equatorial
     * rings based on the provided descriptor. It also calculates and stores the
     * maximum required workspace size (m_work_size).
     *
     * @param descriptor The s2fftDescriptor containing the parameters for the FFT.
     * @return HRESULT indicating success or failure.
     */
    HRESULT Initialize(const s2fftDescriptor &descriptor);

    /**
     * @brief Executes the forward Spherical Harmonic Transform.
     *
     * This method performs the forward FFT operations on the input data
     * across polar and equatorial rings using the pre-initialized cuFFT plans.
     *
     * @param desc The s2fftDescriptor for the current transform.
     * @param stream The CUDA stream to use for execution.
     * @param data Pointer to the input/output data on the device.
     * @param workspace Pointer to the workspace memory on the device.
     * @return HRESULT indicating success or failure.
     */
    HRESULT Forward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data, Complex *workspace);

    /**
     * @brief Executes the backward Spherical Harmonic Transform.
     *
     * This method performs the inverse FFT operations on the input data
     * across polar and equatorial rings using the pre-initialized cuFFT plans.
     *
     * @param desc The s2fftDescriptor for the current transform.
     * @param stream The CUDA stream to use for execution.
     * @param data Pointer to the input/output data on the device.
     * @param workspace Pointer to the workspace memory on the device.
     * @return HRESULT indicating success or failure.
     */
    HRESULT Backward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data, Complex *workspace);

public:
    // cuFFT handles for polar and equatorial FFT plans
    std::vector<cufftHandle> m_polar_plans;
    cufftHandle m_equator_plan;
    std::vector<cufftHandle> m_inverse_polar_plans;
    cufftHandle m_inverse_equator_plan;

    // Parameters defining the SHT geometry and data layout
    int m_nside;
    int m_equatorial_ring_num;
    int64 m_total_pixels;
    int64 m_equatorial_offset_start;
    int64 m_equatorial_offset_end;
    std::vector<int64> m_upper_ring_offsets;
    std::vector<int64> m_lower_ring_offsets;
    size_t m_work_size = 0;  // Maximum workspace size required for FFT plans
};

}  // namespace s2fft

namespace std {
/**
 * @brief Custom hash specialization for s2fftDescriptor.
 *
 * This specialization allows s2fftDescriptor objects to be used as keys
 * in `std::unordered_map` by providing a hash function.
 */
template <>
struct hash<s2fft::s2fftDescriptor> {
    std::size_t operator()(const s2fft::s2fftDescriptor &k) const {
        // Combine hash values of individual members
        size_t hash = std::hash<int64_t>()(k.nside) ^ (std::hash<int64_t>()(k.harmonic_band_limit) << 1) ^
                      (std::hash<bool>()(k.reality) << 2) ^ (std::hash<int>()(k.norm) << 3) ^
                      (std::hash<bool>()(k.shift) << 4) ^ (std::hash<bool>()(k.double_precision) << 5);
        return hash;
    }
};
}  // namespace std

#endif  // S2FFT_H