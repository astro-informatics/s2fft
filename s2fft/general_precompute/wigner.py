import numpy as np
from s2fft.wigner import samples
from s2fft.general_precompute import quadrature_jax, resampling_jax
from s2fft.general_precompute import spin_spherical as s2
from s2fft.general_precompute.construct import healpix_phase_shifts
from functools import partial

from jax import jit
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


def inverse(
    flmn: np.ndarray,
    L: int,
    N: int,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flm (np.ndarray): Wigner coefficients with shape :math:`[L, 2L-1, N]`.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

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
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\beta},
        n_{\alpha}, n_{\gamma}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\beta, \alpha,
        \gamma]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.
    """
    phase_shifts = None
    if sampling.lower() == "healpix":
        phase_shifts = healpix_phase_shifts(L, nside, False)

    if method == "numpy":
        return inverse_transform(
            flmn, kernel, L, N, sampling, nside, phase_shifts
        )
    elif method == "jax":
        return inverse_transform_jax(
            flmn, kernel, L, N, sampling, nside, phase_shifts
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmn: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
    phase_shifts: np.ndarray,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[L, 2L-1, N]`.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (np.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    fnab = np.zeros(samples.fnab_shape(L, N, sampling), dtype=np.complex128)
    fnab[..., m_offset:] = np.einsum(
        "...ntlm, ...nlm -> ...ntm",
        kernel,
        np.einsum(
            "...nlm,...l->...nlm",
            flmn,
            np.sqrt((2 * np.arange(L) + 1) / (16 * np.pi**3)),
        ),
        optimize=True,
    )

    if sampling.lower() in ["mw", "mwss", "dh"]:
        fnab = np.fft.ifftshift(fnab, axes=(-1, -3))
        return np.fft.ifft2(fnab, axes=(-1, -3), norm="forward")

    else:
        raise ValueError("Sampling not yet implemented")


@partial(jit, static_argnums=(2, 3, 4, 5))
def inverse_transform_jax(
    flmn: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
    phase_shifts: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flmn (jnp.ndarray): Wigner coefficients with shape :math:`[L, 2L-1, N]`.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (jnp.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    fnab = jnp.zeros(samples.fnab_shape(L, N, sampling), dtype=jnp.complex128)
    fnab = fnab.at[..., m_offset:].set(
        jnp.einsum(
            "...ntlm, ...nlm -> ...ntm",
            kernel,
            jnp.einsum(
                "...nlm,...l->...nlm",
                flmn,
                jnp.sqrt((2 * jnp.arange(L) + 1) / (16 * jnp.pi**3)),
            ),
            optimize=True,
        )
    )

    if sampling.lower() in ["mw", "mwss", "dh"]:
        fnab = jnp.fft.ifftshift(fnab, axes=(-1, -3))
        return jnp.fft.ifft2(fnab, axes=(-1, -3), norm="forward")

    else:
        raise ValueError("Sampling not yet implemented")


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\beta}, n_{\alpha}, n_{\gamma}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

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
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\beta, \alpha,
        \gamma]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.
    """
    phase_shifts = None
    if sampling.lower() == "healpix":
        phase_shifts = healpix_phase_shifts(L, nside, True)

    if method == "numpy":
        return forward_transform(f, kernel, L, N, sampling, nside, phase_shifts)
    elif method == "jax":
        return forward_transform_jax(
            f, kernel, L, N, sampling, nside, phase_shifts
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
    phase_shifts: np.ndarray,
) -> np.ndarray:
    r"""Compute the forward spherical harmonic transform via precompute (vectorized
    implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (np.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)
    fnab = np.fft.fftshift(np.fft.fft(f, axis=0, norm="backward"), axes=0)
    fnab *= 2 * np.pi / (2 * N - 1)

    for n in range(-N + 1, N):
        flmn[N - 1 + n] = (-1) ** n * s2.forward_transform(
            fnab[N - 1 + n],
            kernel[N - 1 + n],
            L,
            sampling,
            -n,
            nside,
            phase_shifts,
        )

    flmn = np.einsum(
        "nlm,l->nlm", flmn, np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))
    )

    return flmn


@partial(jit, static_argnums=(2, 3, 4, 5))
def forward_transform_jax(
    f: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
    phase_shifts: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the forward spherical harmonic transform via precompute (vectorized
    implementation).

    Args:
        f (jnp.ndarray): Signal on the sphere.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        phase_shifts (jnp.ndarray): Array of ring phase shifts. Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    fban = jnp.fft.fftshift(jnp.fft.fft(f, axis=0, norm="backward"), axes=0)
    fban *= 2 * jnp.pi / (2 * N - 1)

    spins = -jnp.arange(-N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling_jax.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling_jax.upsample_by_two_mwss(fban, L, spins)

    weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    fban = jnp.fft.fft(fban, axis=-1, norm="backward")
    fban = jnp.fft.fftshift(fban, axes=-1)[:, :, m_offset:]

    fban = jnp.einsum("...ntm, ...t->...ntm", fban, weights, optimize=True)
    fban = jnp.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban, optimize=True)
    fban = jnp.einsum(
        "nlm,l->nlm",
        fban,
        jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(L) + 1)),
        optimize=True,
    )

    return fban
