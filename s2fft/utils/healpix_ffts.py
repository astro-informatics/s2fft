from jax import jit

import numpy as np
import jax.numpy as jnp
from s2fft.sampling import s2_samples as samples
from functools import partial


def spectral_folding(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Folds higher frequency Fourier coefficients back onto lower frequency
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
    """Folds higher frequency Fourier coefficients back onto lower frequency
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


def spectral_periodic_extension(
    fm: np.ndarray, nphi: int, L: int
) -> np.ndarray:
    """Extends lower frequency Fourier coefficients onto higher frequency
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
    """Extends lower frequency Fourier coefficients onto higher frequency
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


def healpix_fft(
    f: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
    """Wrapper function for the Forward Fast Fourier Transform with spectral
    back-projection in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    if method.lower() == "numpy":
        return healpix_fft_numpy(f, L, nside, reality)
    elif method.lower() == "jax":
        return healpix_fft_jax(f, L, nside, reality)
    else:
        raise ValueError(f"Method {method} not recognised.")


def healpix_fft_numpy(
    f: np.ndarray, L: int, nside: int, reality: bool
) -> np.ndarray:
    """Computes the Forward Fast Fourier Transform with spectral back-projection
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
def healpix_fft_jax(
    f: jnp.ndarray, L: int, nside: int, reality: bool
) -> jnp.ndarray:
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
    ntheta = samples.ntheta(L, "healpix", nside)
    index = 0
    ftm_rows = []
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        if reality and nphi == 2 * L:
            fm_chunk = jnp.zeros(nphi, dtype=jnp.complex128)
            fm_chunk = fm_chunk.at[nphi // 2 :].set(
                jnp.fft.rfft(
                    jnp.real(f[index : index + nphi]), norm="backward"
                )[:-1]
            )
        else:
            fm_chunk = jnp.fft.fftshift(
                jnp.fft.fft(f[index : index + nphi], norm="backward")
            )
        ftm_rows.append(spectral_periodic_extension_jax(fm_chunk, L))
        index += nphi
    return jnp.stack(ftm_rows)


def healpix_ifft(
    ftm: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
    """Wrapper function for the Inverse Fast Fourier Transform with spectral folding
    in the polar regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
    assert L >= 2 * nside
    if method.lower() == "numpy":
        return healpix_ifft_numpy(ftm, L, nside, reality)
    elif method.lower() == "jax":
        return healpix_ifft_jax(ftm, L, nside, reality)
    else:
        raise ValueError(f"Method {method} not recognised.")


def healpix_ifft_numpy(
    ftm: np.ndarray, L: int, nside: int, reality: bool
) -> np.ndarray:
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
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
    f = np.zeros(
        samples.f_shape(sampling="healpix", nside=nside), dtype=np.complex128
    )
    ntheta = ftm.shape[0]
    index = 0
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = (
            ftm[t] if nphi == 2 * L else spectral_folding(ftm[t], nphi, L)
        )
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
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
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
    f = jnp.zeros(
        samples.f_shape(sampling="healpix", nside=nside), dtype=jnp.complex128
    )
    ntheta = ftm.shape[0]
    index = 0

    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = (
            ftm[t] if nphi == 2 * L else spectral_folding_jax(ftm[t], nphi, L)
        )
        if reality and nphi == 2 * L:
            f = f.at[index : index + nphi].set(
                jnp.fft.irfft(fm_chunk[nphi // 2 :], nphi, norm="forward")
            )
        else:
            f = f.at[index : index + nphi].set(
                jnp.conj(
                    jnp.fft.fft(
                        jnp.fft.ifftshift(jnp.conj(fm_chunk)), norm="backward"
                    )
                )
            )

        index += nphi
    return f


def p2phi_rings(t: np.ndarray, nside: int) -> np.ndarray:
    r"""Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
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
    tt = np.where(
        t + 1 > 3 * nside, shift * np.pi / (2 * (4 * nside - t - 1)), tt
    )
    tt = np.where(t + 1 < nside, shift * np.pi / (2 * (t + 1)), tt)
    return tt


@partial(jit, static_argnums=(1))
def p2phi_rings_jax(t: jnp.ndarray, nside: int) -> jnp.ndarray:
    r"""Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
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
    tt = jnp.where(
        t + 1 > 3 * nside, shift * jnp.pi / (2 * (4 * nside - t - 1)), tt
    )
    tt = jnp.where(t + 1 < nside, shift * jnp.pi / (2 * (t + 1)), tt)
    return tt


def ring_phase_shifts_hp(
    L: int, nside: int, forward: bool = False, reality: bool = False
) -> np.ndarray:
    r"""Generates a phase shift vector for HEALPix for all :math:`\theta` rings.

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
    r"""Generates a phase shift vector for HEALPix for all :math:`\theta` rings. JAX
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
    exponent = jnp.einsum(
        "t, m->tm", phi_offsets, jnp.arange(m_start_ind, L), optimize=True
    )
    return jnp.exp(sign * 1j * exponent)
