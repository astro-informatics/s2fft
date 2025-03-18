# /Users/mun/ONGOING/JASON_2025_Projects/KURTIS/s2fft/s2fft/utils

import numpy as np
import numpy.fft as fft

from s2fft.sampling import s2_samples as samples

import jax.numpy as jnp
import jax



def quad_weights_transform(
    L: int, sampling: str = "mwss", spin: int = 0, nside: int = 0
) -> np.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration *to use in transform* for various sampling schemes.

    Quadrature weights to use in transform for MWSS correspond to quadrature weights
    are twice the base resolution, i.e. 2 * L.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mwss", "dh", "gl", "CK", "healpix}.  Defaults to "mwss".

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        np.ndarray: Quadrature weights *to use in transform* for sampling scheme for
        each :math:`\theta` (weights are identical as :math:`\phi` varies for given
        :math:`\theta`).

    """
    if sampling.lower() == "mwss":
        return quad_weights_mwss_theta_only(2 * L, spin=0) * 2 * np.pi / (2 * L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "ck":
        return quad_weights_ck(L)
    

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")
    


def quad_weights(
    L: int = None, sampling: str = "mw", spin: int = 0, nside: int = None
) -> np.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration for various sampling schemes.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        np.ndarray: Quadrature weights for sampling scheme for each :math:`\theta`
        (weights are identical as :math:`\phi` varies for given :math:`\theta`).

    """
    if sampling.lower() == "mw":
        return quad_weights_mw(L, spin)

    elif sampling.lower() == "mwss":
        return quad_weights_mwss(L, spin)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)
    
    elif sampling.lower() == "ck":
        return quad_weights_ck(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


def quad_weights_hp(nside: int) -> np.ndarray:
    r"""
    Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi`
    integration.

    Note:
        HEALPix weights are identical for all pixels.  Nevertheless, an array of
        weights is returned (with identical values) for consistency of interface
        across other sampling schemes.

    Args:
        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (all weights in array are
        identical).

    """
    npix = 12 * nside**2
    rings = samples.ntheta(sampling="healpix", nside=nside)
    hp_weights = np.zeros(rings, dtype=np.float64)
    hp_weights[:] = 4 * np.pi / npix

    return hp_weights


def quad_weights_gl(L: int) -> np.ndarray:
    r"""
    Compute GL quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    x1, x2 = -1.0, 1.0
    ntheta = samples.ntheta(L, "gl")
    weights = np.zeros(ntheta, dtype=np.float64)

    m = int((L + 1) / 2)
    x1 = 0.5 * (x2 - x1)

    i = np.arange(1, m + 1)
    z = np.cos(np.pi * (i - 0.25) / (L + 0.5))
    z1 = 2.0
    while np.max(np.abs(z - z1)) > 1e-14:
        p1 = 1.0
        p2 = 0.0
        for j in range(1, L + 1):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
        pp = L * (z * p1 - p2) / (z * z - 1.0)
        z1 = z
        z = z1 - p1 / pp

    weights[i - 1] = 2.0 * x1 / ((1.0 - z**2) * pp * pp)
    weights[L + 1 - i - 1] = weights[i - 1]

    return weights * 2 * np.pi / (2 * L - 1)


def quad_weights_dh(L: int) -> np.ndarray:
    r"""
    Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * np.pi / (2 * L - 1)


def quad_weight_dh_theta_only(theta: float, L: int) -> float:
    r"""
    Compute DH quadrature weight for :math:`\theta` integration (only), for given
    :math:`\theta`.

    Args:
        theta (float): :math:`\theta` angle for which to compute weight.

        L (int): Harmonic band-limit.

    Returns:
        float: Weight computed for each :math:`\theta`.

    """
    w = 0.0
    for k in range(0, L):
        w += np.sin((2 * k + 1) * theta) / (2 * k + 1)

    w *= 2 / L * np.sin(theta)

    return w


def quad_weights_mw(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_mw_theta_only(L, spin) * 2 * np.pi / (2 * L - 1)


def quad_weights_mwss(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_mwss_theta_only(L, spin) * 2 * np.pi / (2 * L)


def quad_weights_mwss_theta_only(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """
    w = np.zeros(2 * L, dtype=np.complex128)
    # Extra negative m, so logically -el-1 <= m <= el.
    for i in range(-(L - 1) + 1, L + 1):
        w[i + L - 1] = mw_weights(i - 1)

    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L)

    q = wr[: L + 1]

    q[1:L] = q[1:L] + (-1) ** spin * wr[-1:L:-1]

    return q


def quad_weights_mw_theta_only(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """
    w = np.zeros(2 * L - 1, dtype=np.complex128)
    for i in range(-(L - 1), L):
        w[i + L - 1] = mw_weights(i)

    w *= np.exp(-1j * np.arange(-(L - 1), L) * np.pi / (2 * L - 1))
    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]

    q[: L - 1] = q[: L - 1] + (-1) ** spin * wr[-1 : L - 1 : -1]

    return q


def mw_weights(m: int) -> float:
    r"""
    Compute MW weights given as a function of index m.

    MW weights are defined by

    .. math::

        w(m^\prime) = \int_0^\pi \text{d} \theta \sin \theta \exp(i m^\prime\theta),

    which can be computed analytically.

    Args:
        m (int): Harmonic weight index.

    Returns:
        float: MW weight.

    """
    if m == 1:
        return 1j * np.pi / 2

    elif m == -1:
        return -1j * np.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0


def quad_weights_ck(L: int, spin: int = 0) -> np.ndarray:
    
    r"""
    Compute CL quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """

    n=L
    
# def make_clenshaw_curtis_nodes_and_weights(n: int)
#         -> tuple[jnp.ndarray, jnp.ndarray]:
          
    """Nodes and weights of the Clenshaw-Curtis quadrature."""

    

    if n < 1:
        raise ValueError(f"Clenshaw-Curtis order must be at least 1: n = {n}")

    if n == 1:
        return jnp.array([-1, 1]), jnp.array([1, 1])
    

    N = jnp.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r

    # Clenshaw-Curtis nodes
    x = jnp.cos(jnp.arange(0, n + 1) * jnp.pi / n)

    # Clenshaw-Curtis weights
    w = jnp.concatenate([2 / N / (N - 2), 1 / N[-1:], jnp.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    g0: jnp.ndarray[tuple[int, ...], jnp.dtype[np.floating]] = \
                  -np.ones(n)
    g0[r] = g0[r] + n
    g0[m] = g0[m] + n
    g0 = g0 / (n**2 - 1 + (n % 2))
    w = jnp.fft.ifft(w + g0)
    assert jnp.allclose(w.imag, 0)

    wr = w.real
    # return x, jnp.concatenate([wr, wr[:1]])
    return jnp.concatenate([wr, wr[:1]])






%print(a)


