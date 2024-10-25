from functools import partial
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit


def wigner_subset_to_s2(
    flmn: np.ndarray,
    spins: np.ndarray,
    DW: Tuple[np.ndarray, np.ndarray],
    L: int,
    sampling: str = "mw",
) -> np.ndarray:
    r"""
    Transforms an arbitrary subset of Wigner coefficients onto a subset of spin signals
    on the sphere.

        This function takes a collection of spin spherical harmonic coefficients each with
        a different (though not necessarily unique) spin and maps them back to their
        corresponding pixel-space representations. Following this operation one may
        liftn this collection of spin signals to a signal on SO(3) by exploiting the
        correct Mackey functions.

    Args:
        flmn (np.ndarray): Collection of spin spherical harmonic coefficients
            with shape :math:`[batch, n_s, L, 2L-1, channels]`.
        spins (np.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[np.ndarray, np.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        np.ndarray: A collection of spin signals with shape :math:`[batch, n_s, n_\theta, n_\phi, channels]`.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # EXTRACT VARIOUS PRECOMPUTES
    Delta, _ = DW

    # INDEX VALUES
    n_dim = len(spins)
    m_offset = 1 if sampling.lower() == "mwss" else 0
    ntheta = L + 1 if sampling.lower() == "mwss" else L
    theta0 = 0 if sampling.lower() == "mwss" else np.pi / (2 * L - 1)
    xnlm_size = 2 * L if sampling.lower() == "mwss" else 2 * L - 1

    # REUSED ARRAYS
    m = np.arange(-L + 1 - m_offset, L)
    n = -spins

    # Calculate fmna = i^(n-m)\sum_L Delta^l_am Delta^l_an f^l_mn(2l+1)/(8pi^2)
    x = np.zeros(
        (flmn.shape[0], n_dim, xnlm_size, xnlm_size, flmn.shape[-1]), dtype=flmn.dtype
    )
    x[:, :, m_offset:, m_offset:, :] = np.einsum(
        "bnlmc,lam,lan,l->bnamc",
        flmn,
        Delta,
        Delta[:, :, L - 1 + n],
        (2 * np.arange(L) + 1) / (8 * np.pi**2),
    )

    # APPLY SIGN FUNCTION AND PHASE SHIFT
    x = np.einsum(
        "bnamc,m,n,a->bnamc", x, 1j ** (-m), 1j ** (n), np.exp(1j * m * theta0)
    )

    # IFFT OVER THETA AND PHI
    x = np.fft.ifftshift(x, axes=(-3, -2))
    x = np.fft.ifft(x, axis=-3, norm="forward")[:, :, :ntheta, :, :]
    return np.fft.ifft(x, axis=-2, norm="forward")


@partial(jit, static_argnums=(3, 4))
def wigner_subset_to_s2_jax(
    flmn: jnp.ndarray,
    spins: jnp.ndarray,
    DW: Tuple[jnp.ndarray, jnp.ndarray],
    L: int,
    sampling: str = "mw",
) -> jnp.ndarray:
    r"""
    Transforms an arbitrary subset of Wigner coefficients onto a subset of spin signals
    on the sphere (JAX).

        This function takes a collection of spin spherical harmonic coefficients each with
        a different (though not necessarily unique) spin and maps them back to their
        corresponding pixel-space representations. Following this operation one may
        liftn this collection of spin signals to a signal on SO(3) by exploiting the
        correct Mackey functions.

    Args:
        flmn (jnp.ndarray): Collection of spin spherical harmonic coefficients
            with shape :math:`[batch, n_s, L, 2L-1, channels]`.
        spins (jnp.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[jnp.ndarray, jnp.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        jnp.ndarray: A collection of spin signals with shape :math:`[batch, n_s, n_\theta, n_\phi, channels]`.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # EXTRACT VARIOUS PRECOMPUTES
    Delta, _ = DW

    # INDEX VALUES
    n_dim = len(spins)
    m_offset = 1 if sampling.lower() == "mwss" else 0
    ntheta = L + 1 if sampling.lower() == "mwss" else L
    theta0 = 0 if sampling.lower() == "mwss" else jnp.pi / (2 * L - 1)
    xnlm_size = 2 * L if sampling.lower() == "mwss" else 2 * L - 1

    # REUSED ARRAYS
    m = jnp.arange(-L + 1 - m_offset, L)
    n = -spins

    # Calculate fmna = i^(n-m)\sum_L Delta^l_am Delta^l_an f^l_mn(2l+1)/(8pi^2)
    x = jnp.zeros(
        (flmn.shape[0], n_dim, xnlm_size, xnlm_size, flmn.shape[-1]), dtype=flmn.dtype
    )
    x = x.at[:, :, m_offset:, m_offset:, :].set(
        jnp.einsum(
            "bnlmc,lam,lan,l->bnamc",
            flmn,
            Delta,
            Delta[:, :, L - 1 + n],
            (2 * jnp.arange(L) + 1) / (8 * jnp.pi**2),
        )
    )

    # APPLY SIGN FUNCTION AND PHASE SHIFT
    x = jnp.einsum(
        "bnamc,m,n,a->bnamc", x, 1j ** (-m), 1j ** (n), jnp.exp(1j * m * theta0)
    )

    # IFFT OVER THETA AND PHI
    x = jnp.fft.ifftshift(x, axes=(-3, -2))
    x = jnp.fft.ifft(x, axis=-3, norm="forward")[:, :, :ntheta, :, :]
    return jnp.fft.ifft(x, axis=-2, norm="forward")


def so3_to_wigner_subset(
    f: np.ndarray,
    spins: np.ndarray,
    DW: Tuple[np.ndarray, np.ndarray],
    L: int,
    N: int,
    sampling: str = "mw",
) -> np.ndarray:
    r"""
    Transforms a signal on the rotation group to an arbitrary subset of its Wigner
    coefficients.

        This function takes a signal on the rotation group SO(3) and computes a subset of
        spin spherical harmonic coefficients corresponding to slices across the requested
        spin numbers. These spin numbers can be arbitrarily chosen such that their absolute
        value is less than or equal to the azimuthal band-limit :math:`N\leq L`.

    Args:
        f (np.ndarray): Signal on the rotation group with shape :math:`[batch, n_\gamma, n_\theta,n_\phi, channels]`.
        spins (np.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[np.ndarray, np.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        np.ndarray: Collection of spin spherical harmonic coefficients with shape :math:`[batch, n_s, L, 2L-1, channels]`.

    """
    # COMPUTE FFT OVER GAMMA
    x = np.fft.fft(f, axis=-4, norm="forward")
    x = np.fft.fftshift(x, axes=-4)

    # EXTRACT REQUESTED SPIN COMPONENTS
    x = x[:, N - 1 - spins]

    return s2_to_wigner_subset(x, spins, DW, L, sampling)


@partial(jit, static_argnums=(3, 4, 5))
def so3_to_wigner_subset_jax(
    f: jnp.ndarray,
    spins: jnp.ndarray,
    DW: Tuple[jnp.ndarray, jnp.ndarray],
    L: int,
    N: int,
    sampling: str = "mw",
) -> jnp.ndarray:
    r"""
    Transforms a signal on the rotation group to an arbitrary subset of its Wigner
    coefficients (JAX).

        This function takes a signal on the rotation group SO(3) and computes a subset of
        spin spherical harmonic coefficients corresponding to slices across the requested
        spin numbers. These spin numbers can be arbitrarily chosen such that their absolute
        value is less than or equal to the azimuthal band-limit :math:`N\leq L`.

    Args:
        f (jnp.ndarray): Signal on the rotation group with shape :math:`[batch, n_\gamma, n_\theta,n_\phi, channels]`.
        spins (jnp.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[jnp.ndarray, jnp.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        jnp.ndarray: Collection of spin spherical harmonic coefficients
            with shape :math:`[batch, n_s, L, 2L-1, channels]`.

    """
    # COMPUTE FFT OVER GAMMA
    x = jnp.fft.fft(f, axis=-4, norm="forward")
    x = jnp.fft.fftshift(x, axes=-4)

    # EXTRACT REQUESTED SPIN COMPONENTS
    x = x[:, N - 1 - spins]

    return s2_to_wigner_subset_jax(x, spins, DW, L, sampling)


def s2_to_wigner_subset(
    fs: np.ndarray,
    spins: np.ndarray,
    DW: Tuple[np.ndarray, np.ndarray],
    L: int,
    sampling: str = "mw",
) -> np.ndarray:
    r"""
    Transforms from a collection of arbitrary spin signals on the sphere to the
    corresponding collection of their harmonic coefficients.

        This function takes a multimodal collection of spin spherical harmonic signals
        on the sphere and transforms them into their spin spherical harmonic coefficients.
        These cofficients may then be combined into a subset of Wigner coefficients for
        downstream analysis. In this way one may combine input features across a variety
        of spins into a unified representation.

    Args:
        fs (np.ndarray): Collection of spin signal maps on the sphere with shape :math:`[batch, n_s, n_\theta,n_\phi, channels]`.
        spins (np.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[np.ndarray, np.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        np.ndarray: Collection of spin spherical harmonic coefficients with shape :math:`[batch, n_s, L, 2L-1, channels]`.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # EXTRACT VARIOUS PRECOMPUTES
    Delta, Quads = DW

    # INDEX VALUES & DEFINE PADDING DIMENSIONS
    m_offset = 1 if sampling.lower() == "mwss" else 0
    lpad = (L - 2) if sampling.lower() == "mwss" else (L - 1)
    padims = ((0, 0), (0, 0), (lpad, L - 1), (0, 0), (0, 0))

    # REUSED ARRAYS
    m = np.arange(-L + 1, L)
    n = -spins

    # COMPUTE FFT OVER PHI
    x = np.fft.fft(fs, axis=-2, norm="forward")
    x = np.fft.fftshift(x, axes=-2)[:, :, :, m_offset:, :]

    # PERIODICALLY EXTEND THETA FROM [0,pi]->[0,2pi)
    temp = np.einsum("bntmc,m,n->bntmc", x, (-1) ** np.abs(m), (-1) ** np.abs(n))
    x = np.concatenate((x, np.flip(temp, axis=-3)[:, :, 1:L]), axis=-3)

    # COMPUTE BETA FFT OVER PERIODICALLY EXTENDED FTM
    x = np.fft.fft(x, axis=-3, norm="forward")
    x = np.fft.fftshift(x, axes=-3)

    # APPLY PHASE SHIFT
    if sampling.lower() == "mw":
        x = np.einsum("bntmc,t->bntmc", x, np.exp(-1j * m * np.pi / (2 * L - 1)))

    # FOURIER UPSAMPLE TO 4L-3
    x = np.pad(x, padims)
    x = np.fft.ifftshift(x, axes=-3)
    x = np.fft.ifft(x, axis=-3, norm="forward")

    # PERFORM QUADRATURE CONVOLUTION AS FFT REWEIGHTING IN REAL SPACE
    # NB: Our convention here is conjugate to that of SSHT, in which
    # the weights are conjugate but applied flipped and therefore are
    # equivalent. To avoid flipping here he simply conjugate the weights.
    x = np.einsum("bntmc,t->bntmc", x, Quads)

    # COMPUTE GMM BY FFT
    x = np.fft.fft(x, axis=-3, norm="forward")
    x = np.fft.fftshift(x, axes=-3)[:, :, L - 1 : 3 * L - 2]

    # Calculate flmn = i^(n-m)\sum_t Delta^l_tm Delta^l_tn G_mnt
    x = np.einsum("bnamc,lam,lan->bnlmc", x, Delta, Delta[:, :, L - 1 + n])
    x = np.einsum("bnlmc,m,n->bnlmc", x, 1j ** (m), 1j ** (-n))

    return x * (2.0 * np.pi) ** 2


@partial(jit, static_argnums=(3, 4))
def s2_to_wigner_subset_jax(
    fs: jnp.ndarray,
    spins: jnp.ndarray,
    DW: Tuple[jnp.ndarray, jnp.ndarray],
    L: int,
    sampling: str = "mw",
) -> jnp.ndarray:
    r"""
    Transforms from a collection of arbitrary spin signals on the sphere to the
    corresponding collection of their harmonic coefficients (JAX).

        This function takes a multimodal collection of spin spherical harmonic signals
        on the sphere and transforms them into their spin spherical harmonic coefficients.
        These cofficients may then be combined into a subset of Wigner coefficients for
        downstream analysis. In this way one may combine input features across a variety
        of spins into a unified representation.

    Args:
        fs (jnp.ndarray): Collection of spin signal maps on the sphere with shape :math:`[batch, n_s, n_\theta,n_\phi, channels]`.
        spins (jnp.ndarray): Spins of each field in fs with shape :math:`[n_s]`.
        DW (Tuple[jnp.ndarray, jnp.ndarray]): Fourier coefficients of the reduced
            Wigner d-functions and the corresponding upsampled quadrature weights.
        L (int): Harmonic band-limit.
        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}. Defaults to "mw".

    Returns:
        jnp.ndarray: Collection of spin spherical harmonic coefficients with shape :math:`[batch, n_s, L, 2L-1, channels]`.

    """
    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            f"Fourier-Wigner algorithm does not support {sampling} sampling."
        )

    # EXTRACT VARIOUS PRECOMPUTES
    Delta, Quads = DW

    # INDEX VALUES & DEFINE PADDING DIMENSIONS
    m_offset = 1 if sampling.lower() == "mwss" else 0
    lpad = (L - 2) if sampling.lower() == "mwss" else (L - 1)
    padims = ((0, 0), (0, 0), (lpad, L - 1), (0, 0), (0, 0))

    # REUSED ARRAYS
    m = jnp.arange(-L + 1, L)
    n = -spins

    # COMPUTE FFT OVER PHI
    x = jnp.fft.fft(fs, axis=-2, norm="forward")
    x = jnp.fft.fftshift(x, axes=-2)[:, :, :, m_offset:, :]

    # PERIODICALLY EXTEND THETA FROM [0,pi]->[0,2pi)
    temp = jnp.einsum("bntmc,m,n->bntmc", x, (-1) ** jnp.abs(m), (-1) ** jnp.abs(n))
    x = jnp.concatenate((x, jnp.flip(temp, axis=-3)[:, :, 1:L]), axis=-3)

    # COMPUTE BETA FFT OVER PERIODICALLY EXTENDED FTM
    x = jnp.fft.fft(x, axis=-3, norm="forward")
    x = jnp.fft.fftshift(x, axes=-3)

    # APPLY PHASE SHIFT
    if sampling.lower() == "mw":
        x = jnp.einsum("bntmc,t->bntmc", x, jnp.exp(-1j * m * jnp.pi / (2 * L - 1)))

    # FOURIER UPSAMPLE TO 4L-3
    x = jnp.pad(x, padims)
    x = jnp.fft.ifftshift(x, axes=-3)
    x = jnp.fft.ifft(x, axis=-3, norm="forward")

    # PERFORM QUADRATURE CONVOLUTION AS FFT REWEIGHTING IN REAL SPACE
    # NB: Our convention here is conjugate to that of SSHT, in which
    # the weights are conjugate but applied flipped and therefore are
    # equivalent. To avoid flipping here he simply conjugate the weights.
    x = jnp.einsum("bntmc,t->bntmc", x, Quads)

    # COMPUTE GMM BY FFT
    x = jnp.fft.fft(x, axis=-3, norm="forward")
    x = jnp.fft.fftshift(x, axes=-3)[:, :, L - 1 : 3 * L - 2]

    # Calculate flmn = i^(n-m)\sum_t Delta^l_tm Delta^l_tn G_mnt
    x = jnp.einsum("bnamc,lam,lan->bnlmc", x, Delta, Delta[:, :, L - 1 + n])
    x = jnp.einsum("bnlmc,m,n->bnlmc", x, 1j ** (m), 1j ** (-n))

    return x * (2.0 * jnp.pi) ** 2
