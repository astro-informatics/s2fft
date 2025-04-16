from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from s2fft import recursions
from s2fft.utils import quadrature, quadrature_jax


def inverse_transform(
    flmn: np.ndarray,
    L: int,
    N: int,
    precomps: tuple[np.ndarray, np.ndarray] | None = None,
    reality: bool = False,
    sampling: str = "mw",
) -> np.ndarray:
    """
    Computes the inverse Wigner transform using the Fourier decomposition algorithm.

    Args:
        flmn (np.ndarray): Wigner coefficients.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        precomps (tuple[np.ndarray, np.ndarray], optional): Fourier coefficients of the
            reduced Wigner d-functions and the corresponding upsampled quadrature
            weights. Defaults to None.
        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        np.ndarray: Pixel-space function sampled on the rotation group.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # INDEX VALUES
    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1
    m_offset = 1 if sampling.lower() == "mwss" else 0
    ntheta = L + 1 if sampling.lower() == "mwss" else L
    theta0 = 0 if sampling.lower() == "mwss" else np.pi / (2 * L - 1)
    xnlm_size = 2 * L if sampling.lower() == "mwss" else 2 * L - 1

    # REUSED ARRAYS
    m = np.arange(-L + 1 - m_offset, L)
    n = np.arange(n_start_ind - N + 1, N)

    # Calculate fmna = i^(n-m)\sum_L delta^l_am delta^l_an f^l_mn(2l+1)/(8pi^2)
    x = np.zeros((xnlm_size, xnlm_size, n_dim), dtype=flmn.dtype)
    flmn = np.einsum("nlm,l->nlm", flmn, (2 * np.arange(L) + 1) / (8 * np.pi**2))

    # PRECOMPUTE TRANSFORM
    if precomps is not None:
        delta, _ = precomps
        x = np.zeros((xnlm_size, xnlm_size, n_dim), dtype=flmn.dtype)
        x[m_offset:, m_offset:] = np.einsum(
            "nlm,lam,lan->amn", flmn[n_start_ind:], delta, delta[:, :, L - 1 + n]
        )

    # OTF TRANSFORM
    else:
        delta_el = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
        for el in range(L):
            delta_el = recursions.risbo.compute_full(delta_el, np.pi / 2, L, el)
            x[m_offset:, m_offset:] += np.einsum(
                "nm,am,an->amn",
                flmn[n_start_ind:, el],
                delta_el,
                delta_el[:, L - 1 + n],
            )

    # APPLY SIGN FUNCTION AND PHASE SHIFT
    x = np.einsum("amn,m,n,a->nam", x, 1j ** (-m), 1j ** (n), np.exp(1j * m * theta0))

    # PERFORM FFT OVER BETA, GAMMA, ALPHA
    if reality:
        x = np.fft.ifftshift(x, axes=(1, 2))
        x = np.fft.ifft(x, axis=1, norm="forward")[:, :ntheta]
        x = np.fft.ifft(x, axis=2, norm="forward")
        return np.fft.irfft(x, 2 * N - 1, axis=0, norm="forward")
    else:
        x = np.fft.ifftshift(x)
        x = np.fft.ifft(x, axis=1, norm="forward")[:, :ntheta]
        return np.fft.ifft2(x, axes=(0, 2), norm="forward")


@partial(jit, static_argnums=(1, 2, 4, 5))
def inverse_transform_jax(
    flmn: jnp.ndarray,
    L: int,
    N: int,
    precomps: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    reality: bool = False,
    sampling: str = "mw",
) -> jnp.ndarray:
    """
    Computes the inverse Wigner transform using the Fourier decomposition algorithm (JAX).

    Args:
        flmn (jnp.ndarray): Wigner coefficients.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        precomps (tuple[np.ndarray, np.ndarray], optional): Fourier coefficients of the
            reduced Wigner d-functions and the corresponding upsampled quadrature
            weights. Defaults to None.
        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        jnp.ndarray: Pixel-space function sampled on the rotation group.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # INDEX VALUES
    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1
    m_offset = 1 if sampling.lower() == "mwss" else 0
    ntheta = L + 1 if sampling.lower() == "mwss" else L
    theta0 = 0 if sampling.lower() == "mwss" else jnp.pi / (2 * L - 1)
    xnlm_size = 2 * L if sampling.lower() == "mwss" else 2 * L - 1

    # REUSED ARRAYS
    m = jnp.arange(-L + 1 - m_offset, L)
    n = jnp.arange(n_start_ind - N + 1, N)

    # Calculate fmna = i^(n-m)\sum_L delta^l_am delta^l_an f^l_mn(2l+1)/(8pi^2)
    x = jnp.zeros((xnlm_size, xnlm_size, n_dim), dtype=jnp.complex128)
    flmn = jnp.einsum("nlm,l->nlm", flmn, (2 * jnp.arange(L) + 1) / (8 * jnp.pi**2))

    # PRECOMPUTE TRANSFORM
    if precomps is not None:
        delta, _ = precomps
        x = x.at[m_offset:, m_offset:].set(
            jnp.einsum(
                "nlm,lam,lan->amn", flmn[n_start_ind:], delta, delta[:, :, L - 1 + n]
            )
        )

    # OTF TRANSFORM
    else:
        delta_el = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
        for el in range(L):
            delta_el = recursions.risbo_jax.compute_full(delta_el, jnp.pi / 2, L, el)
            x = x.at[m_offset:, m_offset:].add(
                jnp.einsum(
                    "nm,am,an->amn",
                    flmn[n_start_ind:, el],
                    delta_el,
                    delta_el[:, L - 1 + n],
                )
            )

    # APPLY SIGN FUNCTION AND PHASE SHIFT
    x = jnp.einsum("amn,m,n,a->nam", x, 1j ** (-m), 1j ** (n), jnp.exp(1j * m * theta0))

    # PERFORM FFT OVER BETA, GAMMA, ALPHA
    if reality:
        x = jnp.fft.ifftshift(x, axes=(1, 2))
        x = jnp.fft.ifft(x, axis=1, norm="forward")[:, :ntheta]
        x = jnp.fft.ifft(x, axis=2, norm="forward")
        return jnp.fft.irfft(x, 2 * N - 1, axis=0, norm="forward")
    else:
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.ifft(x, axis=1, norm="forward")[:, :ntheta]
        return jnp.fft.ifft2(x, axes=(0, 2), norm="forward")


def forward_transform(
    f: np.ndarray,
    L: int,
    N: int,
    precomps: tuple[np.ndarray, np.ndarray] | None = None,
    reality: bool = False,
    sampling: str = "mw",
) -> np.ndarray:
    """
    Computes the forward Wigner transform using the Fourier decomposition algorithm.

    Args:
        f (np.ndarray): Function sampled on the rotation group.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        precomps (tuple[np.ndarray, np.ndarray], optional): Fourier coefficients of the
            reduced Wigner d-functions and the corresponding upsampled quadrature
            weights. Defaults to None.
        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        np.ndarray: Wigner coefficients of function f.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # INDEX VALUES
    n_start_ind = N - 1 if reality else 0
    m_offset = 1 if sampling.lower() == "mwss" else 0
    lpad = (L - 2) if sampling.lower() == "mwss" else (L - 1)

    # REUSED ARRAYS
    m = np.arange(-L + 1, L)
    n = np.arange(n_start_ind - N + 1, N)

    # COMPUTE ALPHA + GAMMA FFT
    if reality:
        x = np.fft.rfft(np.real(f), axis=0, norm="forward")
        x = np.fft.fft(x, axis=2, norm="forward")
        x = np.fft.fftshift(x, axes=2)[:, :, m_offset:]
    else:
        x = np.fft.fft2(f, axes=(0, 2), norm="forward")
        x = np.fft.fftshift(x, axes=(0, 2))[:, :, m_offset:]

    # PERIODICALLY EXTEND BETA FROM [0,pi]->[0,2pi)
    temp = np.einsum("ntm,m,n->ntm", x, (-1) ** np.abs(m), (-1) ** np.abs(n))
    x = np.concatenate((x, np.flip(temp, axis=1)[:, 1:L]), axis=1)

    # COMPUTE BETA FFT OVER PERIODICALLY EXTENDED FTM
    x = np.fft.fft(x, axis=1, norm="forward")
    x = np.fft.fftshift(x, axes=1)

    # APPLY PHASE SHIFT
    if sampling.lower() == "mw":
        x = np.einsum("nbm,b->nbm", x, np.exp(-1j * m * np.pi / (2 * L - 1)))

    # FOURIER UPSAMPLE TO 4L-3
    x = np.pad(x, ((0, 0), (lpad, L - 1), (0, 0)))
    x = np.fft.ifftshift(x, axes=1)
    x = np.fft.ifft(x, axis=1, norm="forward")

    # PERFORM QUADRATURE CONVOLUTION AS FFT REWEIGHTING IN REAL SPACE
    # NB: Our convention here is conjugate to that of SSHT, in which
    # the weights are conjugate but applied flipped and therefore are
    # equivalent. To avoid flipping here we simply conjugate the weights.

    if precomps is not None:
        # PRECOMPUTE TRANSFORM
        delta, quads = precomps
    else:
        # OTF TRANSFORM
        delta = None
        # COMPUTE QUADRATURE WEIGHTS
        quads = np.zeros(4 * L - 3, dtype=np.complex128)
        for mm in range(-2 * (L - 1), 2 * (L - 1) + 1):
            quads[mm + 2 * (L - 1)] = quadrature.mw_weights(-mm)
        quads = np.fft.ifft(np.fft.ifftshift(quads), norm="forward")

    # APPLY QUADRATURE
    x = np.einsum("nbm,b->nbm", x, quads)

    # COMPUTE GMM BY FFT
    x = np.fft.fft(x, axis=1, norm="forward")
    x = np.fft.fftshift(x, axes=1)[:, L - 1 : 3 * L - 2]

    # CALCULATE flmn = i^(n-m)\sum_t delta^l_tm delta^l_tn G_mnt
    if delta is not None:
        # PRECOMPUTE TRANSFORM
        x = np.einsum("nam,lam,lan->nlm", x, delta, delta[:, :, L - 1 + n])
    else:
        # OTF TRANSFORM
        delta_el = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
        xx = np.zeros((x.shape[0], L, x.shape[-1]), dtype=x.dtype)
        for el in range(L):
            delta_el = recursions.risbo.compute_full(delta_el, np.pi / 2, L, el)
            xx[:, el] = np.einsum("nam,am,an->nm", x, delta_el, delta_el[:, L - 1 + n])
        x = xx

    x = np.einsum("nbm,m,n->nbm", x, 1j ** (m), 1j ** (-n))

    # SYMMETRY REFLECT FOR N < 0
    if reality:
        temp = np.einsum(
            "nlm,m,n->nlm",
            np.conj(np.flip(x[1:], axis=(-1, -3))),
            (-1) ** np.abs(np.arange(-L + 1, L)),
            (-1) ** np.abs(np.arange(-N + 1, 0)),
        )
        x = np.concatenate((temp, x), axis=0)

    return x * (2.0 * np.pi) ** 2


@partial(jit, static_argnums=(1, 2, 4, 5))
def forward_transform_jax(
    f: jnp.ndarray,
    L: int,
    N: int,
    precomps: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    reality: bool = False,
    sampling: str = "mw",
) -> jnp.ndarray:
    """
    Computes the forward Wigner transform using the Fourier decomposition algorithm (JAX).

    Args:
        f (jnp.ndarray): Function sampled on the rotation group.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        precomps (tuple[np.ndarray, np.ndarray], optional): Fourier coefficients of the
            reduced Wigner d-functions and the corresponding upsampled quadrature
            weights. Defaults to None.
        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        jnp.ndarray: Wigner coefficients of function f.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # INDEX VALUES
    n_start_ind = N - 1 if reality else 0
    m_offset = 1 if sampling.lower() == "mwss" else 0
    lpad = (L - 2) if sampling.lower() == "mwss" else (L - 1)

    # REUSED ARRAYS
    m = jnp.arange(-L + 1, L)
    n = jnp.arange(n_start_ind - N + 1, N)

    # COMPUTE ALPHA + GAMMA FFT
    if reality:
        x = jnp.fft.rfft(jnp.real(f), axis=0, norm="forward")
        x = jnp.fft.fft(x, axis=2, norm="forward")
        x = jnp.fft.fftshift(x, axes=2)[:, :, m_offset:]
    else:
        x = jnp.fft.fft2(f, axes=(0, 2), norm="forward")
        x = jnp.fft.fftshift(x, axes=(0, 2))[:, :, m_offset:]

    # PERIODICALLY EXTEND BETA FROM [0,pi]->[0,2pi)
    temp = jnp.einsum("ntm,m,n->ntm", x, (-1) ** jnp.abs(m), (-1) ** jnp.abs(n))
    x = jnp.concatenate((x, jnp.flip(temp, axis=1)[:, 1:L]), axis=1)

    # COMPUTE BETA FFT OVER PERIODICALLY EXTENDED FTM
    x = jnp.fft.fft(x, axis=1, norm="forward")
    x = jnp.fft.fftshift(x, axes=1)

    # APPLY PHASE SHIFT
    if sampling.lower() == "mw":
        x = jnp.einsum("nbm,b->nbm", x, jnp.exp(-1j * m * jnp.pi / (2 * L - 1)))

    # FOURIER UPSAMPLE TO 4L-3
    x = jnp.pad(x, ((0, 0), (lpad, L - 1), (0, 0)))
    x = jnp.fft.ifftshift(x, axes=1)
    x = jnp.fft.ifft(x, axis=1, norm="forward")

    # PERFORM QUADRATURE CONVOLUTION AS FFT REWEIGHTING IN REAL SPACE
    # NB: Our convention here is conjugate to that of SSHT, in which
    # the weights are conjugate but applied flipped and therefore are
    # equivalent. To avoid flipping here we simply conjugate the weights.

    if precomps is not None:
        # PRECOMPUTE TRANSFORM
        delta, quads = precomps
    else:
        # OTF TRANSFORM
        delta = None
        # COMPUTE QUADRATURE WEIGHTS
        quads = jnp.zeros(4 * L - 3, dtype=jnp.complex128)
        for mm in range(-2 * (L - 1), 2 * (L - 1) + 1):
            quads = quads.at[mm + 2 * (L - 1)].set(quadrature_jax.mw_weights(-mm))
        quads = jnp.fft.ifft(jnp.fft.ifftshift(quads), norm="forward")

    # APPLY QUADRATURE
    x = jnp.einsum("nbm,b->nbm", x, quads)

    # COMPUTE GMM BY FFT
    x = jnp.fft.fft(x, axis=1, norm="forward")
    x = jnp.fft.fftshift(x, axes=1)[:, L - 1 : 3 * L - 2]

    # Calculate flmn = i^(n-m)\sum_t delta^l_tm delta^l_tn G_mnt
    if delta is not None:
        # PRECOMPUTE TRANSFORM
        x = jnp.einsum("nam,lam,lan->nlm", x, delta, delta[:, :, L - 1 + n])
    else:
        # OTF TRANSFORM
        delta_el = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
        xx = jnp.zeros((x.shape[0], L, x.shape[-1]), dtype=x.dtype)
        for el in range(L):
            delta_el = recursions.risbo_jax.compute_full(delta_el, jnp.pi / 2, L, el)
            xx = xx.at[:, el].set(
                jnp.einsum("nam,am,an->nm", x, delta_el, delta_el[:, L - 1 + n])
            )
        x = xx

    x = jnp.einsum("nbm,m,n->nbm", x, 1j ** (m), 1j ** (-n))

    # SYMMETRY REFLECT FOR N < 0
    if reality:
        temp = jnp.einsum(
            "nlm,m,n->nlm",
            jnp.conj(jnp.flip(x[1:], axis=(-1, -3))),
            (-1) ** jnp.abs(jnp.arange(-L + 1, L)),
            (-1) ** jnp.abs(jnp.arange(-N + 1, 0)),
        )
        x = jnp.concatenate((temp, x), axis=0)

    return x * (2.0 * jnp.pi) ** 2
