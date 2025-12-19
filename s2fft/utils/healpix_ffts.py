from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

# did not find promote_dtypes_complex outside _src
from jax._src.numpy.util import promote_dtypes_complex
from jax.core import ShapedArray
from jax.interpreters import batching
from s2fft_lib import _s2fft

from s2fft.sampling import s2_samples as samples
from s2fft.utils.jax_primitive import register_primitive
from s2fft.utils.torch_wrapper import wrap_as_torch_function


def spectral_folding(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """
    Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. aliasing high frequencies.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Lower resolution set of aliased Fourier coefficients.

    """
    assert nphi <= 2 * L

    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    ftm_slice = fm[slice_start:slice_stop]

    idx = 1
    while slice_start - idx >= 0:
        ftm_slice[-idx % nphi] += fm[slice_start - idx]
        idx += 1
    idx = 0
    while slice_stop + idx < len(fm):
        ftm_slice[idx % nphi] += fm[slice_stop + idx]
        idx += 1

    return ftm_slice


def spectral_folding_jax(fm: jnp.ndarray, nphi: int, L: int) -> jnp.ndarray:
    """
    Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. aliasing high frequencies. JAX specific implementation of
    :func:`~spectral_folding`.

    Args:
        fm (jnp.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Lower resolution set of aliased Fourier coefficients.

    """
    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    ftm_slice = fm[slice_start:slice_stop]

    ftm_slice = ftm_slice.at[-jnp.arange(1, L - nphi // 2 + 1) % nphi].add(
        fm[slice_start - jnp.arange(1, L - nphi // 2 + 1)]
    )
    return ftm_slice.at[jnp.arange(L - nphi // 2) % nphi].add(
        fm[slice_stop + jnp.arange(L - nphi // 2)]
    )


spectral_folding_torch = wrap_as_torch_function(spectral_folding_jax)


def spectral_periodic_extension(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """
    Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.

    """
    assert nphi <= 2 * L

    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    fm_full = np.zeros(2 * L, dtype=np.complex128)
    fm_full[slice_start:slice_stop] = fm

    idx = 1
    while slice_start - idx >= 0:
        fm_full[slice_start - idx] = fm[-idx % nphi]
        idx += 1
    idx = 0
    while slice_stop + idx < len(fm_full):
        fm_full[slice_stop + idx] = fm[idx % nphi]
        idx += 1

    return fm_full


def spectral_periodic_extension_jax(fm: jnp.ndarray, L: int) -> jnp.ndarray:
    """
    Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on
    :func:`~spectral_periodic_extension`, modified to be JIT-compilable.

    Args:
        fm (jnp.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Higher resolution set of periodic Fourier coefficients.

    """
    nphi = fm.shape[0]
    return jnp.concatenate(
        (
            fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
            fm,
            fm[jnp.arange(L - (nphi + 1) // 2) % nphi],
        )
    )


spectral_periodic_extension_torch = wrap_as_torch_function(
    spectral_periodic_extension_jax
)


def healpix_fft(
    f: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
    """
    Wrapper function for the Forward Fast Fourier Transform with spectral
    back-projection in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax", "torch", "cuda"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax", "torch", "cuda"}.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.

    """
    if method not in _healpix_fft_functions:
        raise ValueError(f"Method {method} not recognised.")
    return _healpix_fft_functions[method](f, L, nside, reality)


def healpix_fft_numpy(f: np.ndarray, L: int, nside: int, reality: bool) -> np.ndarray:
    """
    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.

    """
    index = 0
    ftm = np.zeros(samples.ftm_shape(L, "healpix", nside), dtype=np.complex128)
    ntheta = ftm.shape[0]
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        if reality and nphi == 2 * L:
            fm_chunk = np.zeros(nphi, dtype=np.complex128)
            fm_chunk[nphi // 2 :] = np.fft.rfft(
                np.real(f[index : index + nphi]), norm="backward"
            )[:-1]
        else:
            fm_chunk = np.fft.fftshift(
                np.fft.fft(f[index : index + nphi], norm="backward")
            )
        ftm[t] = (
            fm_chunk
            if nphi == 2 * L
            else spectral_periodic_extension(fm_chunk, nphi, L)
        )
        index += nphi
    return ftm


@partial(jit, static_argnums=(1, 2, 3))
def healpix_fft_jax(f: jnp.ndarray, L: int, nside: int, reality: bool) -> jnp.ndarray:
    """
    Healpix FFT JAX implementation using jax.numpy/numpy stack
    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity. JAX specific
    implementation of :func:`~healpix_fft_numpy`.

    Args:
        f (jnp.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: Array of Fourier coefficients for all latitudes.

    """

    def f_chunks_to_ftm_rows(f_chunks, nphi):
        if reality and nphi == 2 * L:
            fm_chunks = jnp.concatenate(
                (
                    jnp.zeros((f_chunks.shape[0], nphi // 2)),
                    jnp.fft.rfft(jnp.real(f_chunks), norm="backward")[:, :-1],
                ),
                axis=1,
            )
        else:
            fm_chunks = jnp.fft.fftshift(
                jnp.fft.fft(f_chunks, norm="backward"), axes=-1
            )
        return vmap(spectral_periodic_extension_jax, (0, None))(fm_chunks, L)

    # Process f chunks corresponding to pairs of polar theta rings with the same number
    # of phi samples together to reduce size of unrolled traced computational graph
    ftm_rows_polar = []
    start_index, end_index = 0, 12 * nside**2
    for t in range(0, nside - 1):
        nphi = 4 * (t + 1)
        f_chunks = jnp.stack(
            (f[start_index : start_index + nphi], f[end_index - nphi : end_index])
        )
        ftm_rows_polar.append(f_chunks_to_ftm_rows(f_chunks, nphi))
        start_index, end_index = start_index + nphi, end_index - nphi
    ftm_rows_polar = jnp.stack(ftm_rows_polar)
    # Process all f chunks for the equal sized equatorial theta rings together
    nphi = 4 * nside
    f_chunks_equatorial = f[start_index:end_index].reshape((-1, nphi))
    ftm_rows_equatorial = f_chunks_to_ftm_rows(f_chunks_equatorial, nphi)
    # Concatenate Fourier coefficients for all latitudes, reversing second polar set to
    # account for processing order
    return jnp.concatenate(
        (
            ftm_rows_polar[:, 0],
            ftm_rows_equatorial,
            ftm_rows_polar[::-1, 1],
        )
    )


healpix_fft_torch = wrap_as_torch_function(healpix_fft_jax)


def healpix_ifft(
    ftm: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
    """
    Wrapper function for the Inverse Fast Fourier Transform with spectral folding
    in the polar regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax", "torch", "cuda"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax", "torch", "cuda"}.

    Returns:
        np.ndarray: HEALPix pixel-space array.

    """
    assert L >= 2 * nside
    if method not in _healpix_ifft_functions:
        raise ValueError(f"Method {method} not recognised.")
    return _healpix_ifft_functions[method](ftm, L, nside, reality)


def healpix_ifft_numpy(
    ftm: np.ndarray, L: int, nside: int, reality: bool
) -> np.ndarray:
    """
    Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        np.ndarray: HEALPix pixel-space array.

    """
    f = np.zeros(samples.f_shape(sampling="healpix", nside=nside), dtype=np.complex128)
    ntheta = ftm.shape[0]
    index = 0
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = ftm[t] if nphi == 2 * L else spectral_folding(ftm[t], nphi, L)
        if reality and nphi == 2 * L:
            f[index : index + nphi] = np.fft.irfft(
                fm_chunk[nphi // 2 :], nphi, norm="forward"
            )
        else:
            f[index : index + nphi] = np.fft.ifft(
                np.fft.ifftshift(fm_chunk), norm="forward"
            )
        index += nphi
    return f


@partial(jit, static_argnums=(1, 2, 3))
def healpix_ifft_jax(
    ftm: jnp.ndarray, L: int, nside: int, reality: bool
) -> jnp.ndarray:
    """
    Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing, using JAX. JAX specific implementation of
    :func:`~healpix_ifft_numpy`.

    Args:
        ftm (jnp.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: HEALPix pixel-space array.

    """

    def ftm_rows_to_f_chunks(ftm_rows, nphi):
        fm_chunks = (
            ftm_rows
            if nphi == 2 * L
            else vmap(spectral_folding_jax, (0, None, None))(ftm_rows, nphi, L)
        )
        if reality and nphi == 2 * L:
            return jnp.fft.irfft(fm_chunks[:, nphi // 2 :], nphi, norm="forward")
        else:
            return jnp.fft.ifft(jnp.fft.ifftshift(fm_chunks, axes=-1), norm="forward")

    # Process ftm rows corresponding to pairs of polar theta rings with the same number
    # of phi samples together to reduce size of unrolled traced computational graph
    f_chunks_polar = [
        ftm_rows_to_f_chunks(jnp.stack((ftm[t], ftm[-(t + 1)])), 4 * (t + 1))
        for t in range(nside - 1)
    ]
    # Process all ftm rows for the equal sized equatorial theta rings together
    f_chunks_equatorial = ftm_rows_to_f_chunks(ftm[nside - 1 : 3 * nside], 4 * nside)
    # Concatenate f chunks for all theta rings together, reversing second polar set
    # to account for processing order
    return jnp.concatenate(
        [f_chunks_polar[t][0] for t in range(nside - 1)]
        + [f_chunks_equatorial.flatten()]
        + [f_chunks_polar[t][1] for t in reversed(range(nside - 1))]
    )


healpix_ifft_torch = wrap_as_torch_function(healpix_ifft_jax)


def p2phi_rings(t: np.ndarray, nside: int) -> np.ndarray:
    r"""
    Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
    Vectorised implementation of :func:`~samples.p2phi_ring`.

    Args:
        t (np.ndarrray): vector of :math:`\theta` ring indicies, i.e. [0,1,...,ntheta-1]

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: :math:`\phi` offset for each ring index.

    """
    shift = 1 / 2
    tt = np.zeros_like(t)
    tt = np.where(
        (t + 1 >= nside) & (t + 1 <= 3 * nside),
        shift * ((t - nside + 2) % 2) * np.pi / (2 * nside),
        tt,
    )
    tt = np.where(t + 1 > 3 * nside, shift * np.pi / (2 * (4 * nside - t - 1)), tt)
    tt = np.where(t + 1 < nside, shift * np.pi / (2 * (t + 1)), tt)
    return tt


@partial(jit, static_argnums=(1))
def p2phi_rings_jax(t: jnp.ndarray, nside: int) -> jnp.ndarray:
    r"""
    Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
    JAX implementation of :func:`~p2phi_rings`.

    Args:
        t (jnp.ndarrray): vector of :math:`\theta` ring indicies, i.e. [0,1,...,ntheta-1]

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: :math:`\phi` offset for each ring index.

    """
    shift = 1 / 2
    tt = jnp.zeros_like(t)
    tt = jnp.where(
        (t + 1 >= nside) & (t + 1 <= 3 * nside),
        shift * ((t - nside + 2) % 2) * jnp.pi / (2 * nside),
        tt,
    )
    tt = jnp.where(t + 1 > 3 * nside, shift * jnp.pi / (2 * (4 * nside - t - 1)), tt)
    tt = jnp.where(t + 1 < nside, shift * jnp.pi / (2 * (t + 1)), tt)
    return tt


def ring_phase_shifts_hp(
    L: int, nside: int, forward: bool = False, reality: bool = False
) -> np.ndarray:
    r"""
    Generates a phase shift vector for HEALPix for all :math:`\theta` rings.

    Args:
        L (int, optional): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Returns:
        np.ndarray: Vector of phase shifts with shape :math:`[n_{\theta}, 2L-1]`.

    """
    t = np.arange(samples.ntheta(L, "healpix", nside))
    phi_offsets = p2phi_rings(t, nside)
    sign = -1 if forward else 1
    m_start_ind = 0 if reality else -L + 1
    exponent = np.einsum("t, m->tm", phi_offsets, np.arange(m_start_ind, L))
    return np.exp(sign * 1j * exponent)


@partial(jit, static_argnums=(0, 1, 2, 3))
def ring_phase_shifts_hp_jax(
    L: int, nside: int, forward: bool = False, reality: bool = False
) -> jnp.ndarray:
    r"""
    Generates a phase shift vector for HEALPix for all :math:`\theta` rings. JAX
    implementation of :func:`~ring_phase_shifts_hp`.

    Args:
        L (int, optional): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Returns:
        jnp.ndarray: Vector of phase shifts with shape :math:`[n_{\theta}, 2L-1]`.

    """
    t = jnp.arange(samples.ntheta(L, "healpix", nside))
    phi_offsets = p2phi_rings_jax(t, nside)
    sign = -1 if forward else 1
    m_start_ind = 0 if reality else -L + 1
    # Step 5: Calculate the exponent for the phase shifts using JAX einsum.
    exponent = jnp.einsum(
        "t, m->tm", phi_offsets, jnp.arange(m_start_ind, L), optimize=True
    )
    # Step 6: Return the complex exponential of the exponent.
    return jnp.exp(sign * 1j * exponent)


# Custom healpix_fft_cuda primitive


def _get_lowering_info(fft_type, norm, out_dtype):
    # Step 1: Determine if double precision is used based on output dtype.
    if out_dtype == np.complex64:
        is_double = False
    elif out_dtype == np.complex128:
        is_double = True
    else:
        raise ValueError(f"Unknown output type {out_dtype}")

    # Step 2: Determine if it's a forward transform.
    forward = fft_type == "forward"
    # Step 3: Determine if normalization should be applied.
    if (forward and norm == "backward") or (not forward and norm == "forward"):
        normalize = False
    elif (forward and norm == "forward") or (not forward and norm == "backward"):
        normalize = True
    else:
        raise ValueError(f"Unknown norm {norm}")

    # Step 4: Return the determined flags.
    return is_double, forward, normalize


def _healpix_fft_cuda_abstract(f, L, nside, reality, fft_type, norm, adjoint):
    """
    Abstract evaluation for the HEALPix FFT CUDA primitive.
    This function defines the output shapes and dtypes for the JAX primitive.

    Args:
        f: Input array.
        L: Harmonic band-limit.
        nside: HEALPix Nside resolution parameter.
        reality: Whether the signal is real.
        fft_type: Type of FFT ("forward" or "backward").
        norm: Normalization type.
        adjoint: Whether it's an adjoint operation.

    Returns:
        Tuple of ShapedArray objects for output, workspace, and callback parameters.

    """
    # Step 1: Get lowering information (double precision, forward/backward, normalize).
    is_double, forward, normalize = _get_lowering_info(fft_type, norm, f.dtype)

    # Step 2: Determine workspace size and type based on precision.
    if is_double:
        # For double precision, build descriptor for C128 and calculate workspace size.
        worksize = _s2fft.build_descriptor_C128(
            nside, L, reality, forward, normalize, adjoint
        )
        worksize //= 16  # 16 bytes per C128 element
        workspace_shape = (worksize,)
        workspace_dtype = np.complex128
    else:
        # For single precision, build descriptor for C64 and calculate workspace size.
        worksize = _s2fft.build_descriptor_C64(
            nside, L, reality, forward, normalize, adjoint
        )
        worksize //= 8  # 8 bytes per C64 element
        workspace_shape = (worksize,)
        workspace_dtype = np.complex64
    # Step 3: Define output shapes based on FFT type.
    healpix_size = (nside**2 * 12,)
    ftm_size = (4 * nside - 1, 2 * L)
    if fft_type == "forward":
        batch_shape = (f.shape[0],) if f.ndim == 2 else ()
        out_shape = batch_shape + ftm_size
        assert (f.shape[-1],) == healpix_size

    elif fft_type == "backward":
        batch_shape = (f.shape[0],) if f.ndim == 3 else ()
        out_shape = batch_shape + healpix_size
        assert f.shape[-2:] == ftm_size
    else:
        raise ValueError(f"fft_type {fft_type} not recognised.")

    # Step 4: Create ShapedArray objects for output and workspace.
    workspace_aval = ShapedArray(
        shape=batch_shape + workspace_shape, dtype=workspace_dtype
    )

    # Step 5: Return the ShapedArray objects.
    return (
        f.update(shape=out_shape, dtype=f.dtype),
        workspace_aval,
    )


class MissingCUDASupport(Exception):  # noqa : D107
    def __init__(self):  # noqa : D107
        super().__init__("""
                        S2FFT was compiled without CUDA support. Cuda functions are not supported.
                        Please make sure that nvcc is in your path and $CUDA_HOME is set then reinstall s2fft using pip.
                        """)


def _healpix_fft_cuda_lowering(ctx, f, *, L, nside, reality, fft_type, norm, adjoint):
    """
    Lowering rule for the HEALPix FFT CUDA primitive.
    This function translates the JAX primitive call into a call to the underlying CUDA FFI.

    Args:
        ctx: Lowering context.
        f: Input array.
        L: Harmonic band-limit.
        nside: HEALPix Nside resolution parameter.
        reality: Whether the signal is real.
        fft_type: Type of FFT ("forward" or "backward").
        norm: Normalization type.
        adjoint: Whether it's an adjoint operation.

    Returns:
        The result of the FFI call.

    """
    # Step 1: Check if CUDA support is compiled in.
    if not _s2fft.COMPILED_WITH_CUDA:
        raise MissingCUDASupport()

    # Step 2: Get the abstract evaluation results for the outputs.
    (aval_out, _) = ctx.avals_out

    # Step 3: Get lowering information (double precision, forward/backward, normalize).
    is_double, forward, normalize = _get_lowering_info(fft_type, norm, aval_out.dtype)

    # Step 4: Select the appropriate FFI lowering function based on precision.
    if is_double:
        ffi_lowered = jax.ffi.ffi_lowering("healpix_fft_cuda_c128")
    else:
        ffi_lowered = jax.ffi.ffi_lowering("healpix_fft_cuda_c64")

    # Step 5: Call the FFI lowering function with the context and parameters.
    return ffi_lowered(
        ctx,
        f,
        nside=nside,
        harmonic_band_limit=L,
        reality=reality,
        normalize=normalize,
        forward=forward,
        adjoint=adjoint,
    )


def _healpix_fft_cuda_batching_rule(
    batched_args, batched_axis, L, nside, reality, fft_type, norm, adjoint
):
    """
    Batching rule for the HEALPix FFT CUDA primitive.
    This function defines how the primitive behaves under JAX's automatic batching.

    Args:
        batched_args: Tuple of batched arguments.
        batched_axis: Tuple of axes along which arguments are batched.
        L: Harmonic band-limit.
        nside: HEALPix Nside resolution parameter.
        reality: Whether the signal is real.
        fft_type: Type of FFT ("forward" or "backward").
        norm: Normalization type.
        adjoint: Whether it's an adjoint operation.

    Returns:
        Tuple of (output, output_batch_axes).

    """
    # Step 1: Unpack batched arguments and batching axes.
    (x,) = batched_args
    (bd,) = batched_axis

    # Step 2: Assert correct input dimensions based on FFT type.
    if fft_type == "forward":
        assert x.ndim == 2
    elif fft_type == "backward":
        assert x.ndim == 3
    else:
        raise ValueError(f"fft_type {fft_type} not recognised.")

    # Step 3: Move the batching axis to the front.
    x = batching.moveaxis(x, bd, 0)

    # Step 4: Bind the primitive with the batched input.
    out = _healpix_fft_cuda_primitive.bind(
        x,
        L=L,
        nside=nside,
        reality=reality,
        fft_type=fft_type,
        norm=norm,
        adjoint=adjoint,
    )
    # Step 5: Define batching axes for the outputs (all at axis 0).
    batchout = (0,) * len(out)

    # Step 6: Return the output and their batching axes.
    return out, batchout


def _healpix_fft_cuda_transpose(
    df: jnp.ndarray,
    L: int,
    nside: int,
    reality: bool,
    fft_type: str,
    norm: str,
    adjoint: bool,
) -> jnp.ndarray:
    """
    Transpose rule for the HEALPix FFT CUDA primitive.
    This function defines how the adjoint of the primitive is computed for automatic differentiation.

    Args:
        df: Tangent (gradient) of the output.
        L: Harmonic band-limit.
        nside: HEALPix Nside resolution parameter.
        reality: Whether the signal is real.
        fft_type: Type of FFT ("forward" or "backward").
        norm: Normalization type.
        adjoint: Whether it's an adjoint operation.

    Returns:
        The adjoint of the input.

    """
    # Step 1: Invert the FFT type and normalization for the adjoint operation.
    fft_type = "backward" if fft_type == "forward" else "forward"
    norm = "backward" if norm == "forward" else "forward"

    # Step 2: Bind the primitive with the tangent and inverted parameters.
    # Access df[0] as df is a tuple of tangents for multiple outputs.
    # Return [0] as the primitive also returns multiple outputs, and we only need the first one for the adjoint.
    return (
        _healpix_fft_cuda_primitive.bind(
            df[0],
            L=L,
            nside=nside,
            reality=reality,
            fft_type=fft_type,
            norm=norm,
            adjoint=not adjoint,
        )[0],
    )


# Register healpfix_fft_cuda custom call target
for name, fn in _s2fft.registration().items():
    jax.ffi.register_ffi_target(name, fn, platform="CUDA")

# Step 1: Register the HEALPix FFT CUDA primitive with JAX.
_healpix_fft_cuda_primitive = register_primitive(
    "healpix_fft_cuda",
    multiple_results=True,  # Indicates that the primitive returns multiple outputs.
    abstract_evaluation=_healpix_fft_cuda_abstract,
    lowering_per_platform={None: _healpix_fft_cuda_lowering},
    transpose=_healpix_fft_cuda_transpose,
    batcher=_healpix_fft_cuda_batching_rule,
    is_linear=True,
)


@partial(jit, static_argnums=(1, 2, 3))
def healpix_fft_cuda(
    f: jnp.ndarray, L: int, nside: int, reality: bool, norm: str = "backward"
) -> jnp.ndarray:
    """
    Healpix FFT JAX implementation using custom CUDA primitive.

    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (jnp.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: Array of Fourier coefficients for all latitudes.

    """
    # Step 1: Promote input data to complex dtype if necessary.
    (f,) = promote_dtypes_complex(f)
    # Step 2: Bind the input to the CUDA primitive. It returns multiple outputs (out, workspace).
    out, _ = _healpix_fft_cuda_primitive.bind(
        f,
        L=L,
        nside=nside,
        reality=reality,
        fft_type="forward",
        norm=norm,
        adjoint=False,
    )
    # Step 3: Return only the primary output (Fourier coefficients).
    return out


@partial(jit, static_argnums=(1, 2, 3, 4))
def healpix_ifft_cuda(
    ftm: jnp.ndarray, L: int, nside: int, reality: bool, norm: str = "forward"
) -> jnp.ndarray:
    """
    Healpix IFFT JAX implementation using custom CUDA primitive.

    Computes the inverse fast Fourier transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (jnp.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: HEALPix pixel-space array.

    """
    # Step 1: Promote input data to complex dtype if necessary.
    (ftm,) = promote_dtypes_complex(ftm)
    # Step 2: Bind the input to the CUDA primitive. It returns multiple outputs (out, workspace).
    out, _ = _healpix_fft_cuda_primitive.bind(
        ftm,
        L=L,
        nside=nside,
        reality=reality,
        fft_type="backward",
        norm=norm,
        adjoint=False,
    )
    # Step 3: Return only the primary output (pixel-space array).
    return out


_healpix_fft_functions = {
    "numpy": healpix_fft_numpy,
    "jax": healpix_fft_jax,
    "cuda": healpix_fft_cuda,
    "torch": healpix_fft_torch,
}

_healpix_ifft_functions = {
    "numpy": healpix_ifft_numpy,
    "jax": healpix_ifft_jax,
    "cuda": healpix_ifft_cuda,
    "torch": healpix_ifft_torch,
}
