import numpy as np
from s2fft import resampling
from s2fft.wigner import samples
import s2fft.healpix_ffts as hp
from s2fft.general_precompute import resampling_jax
from s2fft.general_precompute.construct import healpix_phase_shifts
from functools import partial

from jax import jit
import jax.numpy as jnp


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
    if method == "numpy":
        return inverse_transform(flmn, kernel, L, N, sampling, nside)
    elif method == "jax":
        return inverse_transform_jax(flmn, kernel, L, N, sampling, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmn: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
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

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    fnab = np.zeros(
        samples.fnab_shape(L, N, sampling, nside), dtype=np.complex64
    )
    fnab[..., m_offset:] = np.einsum(
        "...ntlm, ...nlm -> ...ntm", kernel, flmn, optimize=True
    )

    if sampling.lower() in "healpix":
        f = np.zeros(samples.f_shape(L, N, sampling, nside), dtype=np.complex64)
        for n in range(-N + 1, N):
            ind = N - 1 + n
            f[ind] = hp.healpix_ifft(fnab[ind], L, nside, "numpy")
        return np.fft.ifft(
            np.fft.ifftshift(f, axes=-2), axis=-2, norm="forward"
        )

    else:
        fnab = np.fft.ifftshift(fnab, axes=(-1, -3))
        return np.fft.ifft2(fnab, axes=(-1, -3), norm="forward")


@partial(jit, static_argnums=(2, 3, 4, 5))
def inverse_transform_jax(
    flmn: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
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

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    fnab = jnp.zeros(
        samples.fnab_shape(L, N, sampling, nside), dtype=jnp.complex64
    )
    fnab = fnab.at[..., m_offset:].set(
        jnp.einsum("...ntlm, ...nlm -> ...ntm", kernel, flmn, optimize=True)
    )

    if sampling.lower() in "healpix":
        f = jnp.zeros(
            samples.f_shape(L, N, sampling, nside), dtype=jnp.complex64
        )
        for n in range(-N + 1, N):
            ind = N - 1 + n
            f = f.at[ind].set(hp.healpix_ifft(fnab[ind], L, nside, "jax"))
        return jnp.fft.ifft(
            jnp.fft.ifftshift(f, axes=-2), axis=-2, norm="forward"
        )

    else:
        fnab = jnp.fft.ifftshift(fnab, axes=(-1, -3))
        return jnp.fft.ifft2(fnab, axes=(-1, -3), norm="forward")


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
    if method == "numpy":
        return forward_transform(f, kernel, L, N, sampling, nside)
    elif method == "jax":
        return forward_transform_jax(f, kernel, L, N, sampling, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
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

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    fban = np.fft.fftshift(np.fft.fft(f, axis=0, norm="backward"), axes=0)
    spins = -np.arange(-N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling.upsample_by_two_mwss(fban, L, spins)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() in "healpix":
        temp = np.zeros(
            samples.fnab_shape(L, N, sampling, nside), dtype=np.complex64
        )
        for n in range(-N + 1, N):
            ind = N - 1 + n
            temp[ind] = hp.healpix_fft(fban[ind], L, nside, "numpy")
        fban = temp[..., m_offset:]
    else:
        fban = np.fft.fft(fban, axis=-1, norm="backward")
        fban = np.fft.fftshift(fban, axes=-1)[:, :, m_offset:]

    return np.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban)


@partial(jit, static_argnums=(2, 3, 4, 5))
def forward_transform_jax(
    f: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    nside: int,
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

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    fban = jnp.fft.fftshift(jnp.fft.fft(f, axis=0, norm="backward"), axes=0)
    spins = -jnp.arange(-N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling_jax.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling_jax.upsample_by_two_mwss(fban, L, spins)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() in "healpix":
        temp = jnp.zeros(
            samples.fnab_shape(L, N, sampling, nside), dtype=jnp.complex64
        )
        for n in range(-N + 1, N):
            ind = N - 1 + n
            temp = temp.at[ind].set(hp.healpix_fft(fban[ind], L, nside, "jax"))
        fban = temp[..., m_offset:]
    else:
        fban = jnp.fft.fft(fban, axis=-1, norm="backward")
        fban = jnp.fft.fftshift(fban, axes=-1)[:, :, m_offset:]

    return jnp.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban, optimize=True)
