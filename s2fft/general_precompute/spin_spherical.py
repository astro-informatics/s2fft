import numpy as np
from s2fft import samples, quadrature, resampling
from s2fft.general_precompute import resampling_jax, quadrature_jax
import s2fft.healpix_ffts as hp
from s2fft.general_precompute.construct import healpix_phase_shifts
from functools import partial

from jax import jit
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


def inverse(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the inverse spherical harmonic transform via precompute.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        kernel (np.ndarray, optional): Wigner-d kernel. Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "jax".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients with shape.
    """
    phase_shifts = None
    if sampling.lower() == "healpix":
        phase_shifts = healpix_phase_shifts(L, nside, False)

    if method == "numpy":
        return inverse_transform(
            flm, kernel, L, sampling, spin, nside, phase_shifts
        )
    elif method == "jax":
        return inverse_transform_jax(
            flm, kernel, L, sampling, spin, nside, phase_shifts
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flm: np.ndarray,
    kernel: np.ndarray,
    L: int,
    sampling: str,
    spin: int,
    nside: int,
    phase_shifts: np.ndarray,
) -> np.ndarray:
    r"""Compute the forward spherical harmonic transform via precompute (vectorized
    implementation).

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        spin (int): Harmonic spin.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (np.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    ftm[:, m_offset:] = np.einsum("...tlm, ...lm -> ...tm", kernel, flm)
    ftm *= (-1) ** spin

    if sampling.lower() == "healpix":
        ftm[:, m_offset:] *= phase_shifts
        f = hp.healpix_ifft(ftm, L, nside, "numpy")

    else:
        f = np.fft.ifftshift(ftm, axes=-1)
        f = np.fft.ifft(f, axis=-1, norm="forward")
    return f


@partial(jit, static_argnums=(2, 3, 4, 5))
def inverse_transform_jax(
    flm: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    sampling: str,
    spin: int,
    nside: int,
    phase_shifts: np.ndarray,
) -> jnp.ndarray:
    r"""Compute the inverse spherical harmonic transform via precompute (JAX
    implementation).

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        spin (int): Harmonic spin.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (np.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Pixel-space coefficients with shape.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    ftm = jnp.zeros(samples.ftm_shape(L, sampling, nside), dtype=jnp.complex128)
    ftm = ftm.at[:, m_offset:].add(
        jnp.einsum("...tlm, ...lm -> ...tm", kernel, flm, optimize=True)
    )
    ftm *= (-1) ** spin

    if sampling.lower() == "healpix":
        ftm = ftm.at[:, m_offset:].multiply(phase_shifts)
        f = hp.healpix_ifft(ftm, L, nside, "jax")

    else:
        f = jnp.fft.ifftshift(ftm, axes=-1)
        f = jnp.fft.ifft(f, axis=-1, norm="forward")
    return f


def forward(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the forward spherical harmonic transform via precompute.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        kernel (np.ndarray, optional): Wigner-d kernel. Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "jax".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    phase_shifts = None
    if sampling.lower() == "healpix":
        phase_shifts = healpix_phase_shifts(L, nside, True)

    if method == "numpy":
        return forward_transform(
            f, kernel, L, sampling, spin, nside, phase_shifts
        )
    elif method == "jax":
        return forward_transform_jax(
            f, kernel, L, sampling, spin, nside, phase_shifts
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    kernel: np.ndarray,
    L: int,
    sampling: str,
    spin: int,
    nside: int,
    phase_shifts: np.ndarray,
) -> np.ndarray:
    r"""Compute the forward spherical harmonic transform via precompute (vectorized
    implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        spin (int): Harmonic spin.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (np.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)

    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside, "numpy")[:, m_offset:]
        ftm *= phase_shifts

    else:
        ftm = np.fft.fft(f, axis=-1, norm="backward")
        ftm = np.fft.fftshift(ftm, axes=-1)[:, m_offset:]

    ftm = np.einsum("...tm, ...t->...tm", ftm, weights)
    flm = np.einsum("...tlm, ...tm -> ...lm", kernel, ftm)

    return flm * (-1) ** spin


@partial(jit, static_argnums=(2, 3, 4, 5))
def forward_transform_jax(
    f: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    sampling: str,
    spin: int,
    nside: int,
    phase_shifts: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the forward spherical harmonic tranclearsform via precompute (vectorized
    implementation).

    Args:
        f (jnp.ndarray): Signal on the sphere.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        spin (int): Harmonic spin.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (jnp.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    if sampling.lower() == "mw":
        f = jnp.squeeze(
            resampling_jax.mw_to_mwss(jnp.expand_dims(f, 0), L, spin)
        )

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = jnp.squeeze(
            resampling_jax.upsample_by_two_mwss(jnp.expand_dims(f, 0), L, spin)
        )

    weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside, "jax")[:, m_offset:]
        ftm *= phase_shifts

    else:
        ftm = jnp.fft.fft(f, axis=-1, norm="backward")
        ftm = jnp.fft.fftshift(ftm, axes=-1)[:, m_offset:]

    ftm = jnp.einsum("...tm, ...t->...tm", ftm, weights, optimize=True)
    flm = jnp.einsum("...tlm, ...tm -> ...lm", kernel, ftm, optimize=True)

    return flm * (-1) ** spin
