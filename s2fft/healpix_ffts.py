import numpy as np
import numpy.fft as fft
import s2fft.samples as samples

from jax import jit
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax.config import config

config.update("jax_enable_x64", True)
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
    coefficients, i.e. aliasing high frequencies.

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


def spectral_periodic_extension_jax(fm, L):
    """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on `spectral_periodic_extension`,
    modified to be JIT-compilable.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.
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
    f: np.ndarray, L: int, nside: int, method: str = "numpy"
) -> np.ndarray:
    """Wrapper function for the Forward Fast Fourier Transform with spectral
    back-projection in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    # assert L >= 2 * nside
    if method.lower() == "numpy":
        return healpix_fft_numpy(f, L, nside)
    elif method.lower() == "jax":
        return healpix_fft_jax(f, L, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def healpix_fft_numpy(f: np.ndarray, L: int, nside: int) -> np.ndarray:
    """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    index = 0
    ftm = np.zeros(samples.ftm_shape(L, "healpix", nside), dtype=np.complex128)
    ntheta = ftm.shape[0]
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = fft.fftshift(
            fft.fft(f[index : index + nphi], norm="backward")
        )
        ftm[t] = (
            fm_chunk
            if nphi == 2 * L
            else spectral_periodic_extension(fm_chunk, nphi, L)
        )
        index += nphi
    return ftm


@partial(jit, static_argnums=(1, 2))
def healpix_fft_jax(f: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    """
    Healpix FFT JAX implementation using jax.numpy/numpy stack
    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (jnp.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: Array of Fourier coefficients for all latitudes.
    """
    ntheta = samples.ntheta(L, "healpix", nside)
    index = 0
    ftm_rows = []
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = jnp.fft.fftshift(
            jnp.fft.fft(f[index : index + nphi], norm="backward")
        )
        ftm_rows.append(spectral_periodic_extension_jax(fm_chunk, L))
        index += nphi
    return jnp.stack(ftm_rows)


def healpix_ifft(
    ftm: np.ndarray, L: int, nside: int, method: str = "numpy"
) -> np.ndarray:
    """Wrapper function for the Inverse Fast Fourier Transform with spectral folding
    in the polar regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
    assert L >= 2 * nside
    if method.lower() == "numpy":
        return healpix_ifft_numpy(ftm, L, nside)
    elif method.lower() == "jax":
        return healpix_ifft_jax(ftm, L, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def healpix_ifft_numpy(ftm: np.ndarray, L: int, nside: int) -> np.ndarray:
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

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
        f[index : index + nphi] = fft.ifft(
            fft.ifftshift(fm_chunk), norm="forward"
        )
        index += nphi
    return f


def healpix_ifft_jax(ftm: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing, using JAX.

    Args:
        ftm (jnp.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

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
        f = f.at[index : index + nphi].set(
            jnp.fft.ifft(jnp.fft.ifftshift(fm_chunk), norm="forward")
        )
        index += nphi
    return f
