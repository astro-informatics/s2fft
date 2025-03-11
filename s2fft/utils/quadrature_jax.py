from functools import partial as _partial

import jax
import jax.numpy as jnp
from jax import jit as _jit

from s2fft.sampling import s2_samples as samples


@_partial(_jit, static_argnums=(0, 1, 2))
def quad_weights_transform(
    L: int, sampling: str = "mwss", nside: int = 0
) -> jnp.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration *to use in transform* for various sampling schemes. JAX implementation of
    :func:`~s2fft.quadrature.quad_weights_transform`.

    Quadrature weights to use in transform for MWSS correspond to quadrature weights
    are twice the base resolution, i.e. 2 * L.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mwss", "dh", "gl", "healpix}.  Defaults to "mwss".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        jnp.ndarray: Quadrature weights *to use in transform* for sampling scheme for
        each :math:`\theta` (weights are identical as :math:`\phi` varies for given
        :math:`\theta`).

    """
    if sampling.lower() == "mwss":
        return quad_weights_mwss_theta_only(2 * L) * 2 * jnp.pi / (2 * L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


@_partial(_jit, static_argnums=(0, 1, 2))
def quad_weights(L: int = None, sampling: str = "mw", nside: int = None) -> jnp.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration for various sampling schemes. JAX implementation of
    :func:`~s2fft.quadrature.quad_weights`.

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
        jnp.ndarray: Quadrature weights for sampling scheme for each :math:`\theta`
        (weights are identical as :math:`\phi` varies for given :math:`\theta`).

    """
    if sampling.lower() == "mw":
        return quad_weights_mw(L)

    elif sampling.lower() == "mwss":
        return quad_weights_mwss(L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


@_partial(_jit, static_argnums=(0))
def quad_weights_hp(nside: int) -> jnp.ndarray:
    r"""
    Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi`
    integration. JAX implementation of :func:`s2fft.quadrature.quad_weights_hp`.

    Note:
        HEALPix weights are identical for all pixels.  Nevertheless, an array of
        weights is returned (with identical values) for consistency of interface
        across other sampling schemes.

    Args:
        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (all weights in array are
        identical).

    """
    npix = 12 * nside**2
    rings = samples.ntheta(sampling="healpix", nside=nside)
    return jnp.ones(rings, dtype=jnp.float64) * 4 * jnp.pi / npix


@_partial(_jit, static_argnums=(0))
def quad_weights_gl(L: int) -> jnp.ndarray:
    r"""
    Compute GL quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    x1, x2 = -1.0, 1.0
    ntheta = samples.ntheta(L, "gl")
    weights = jnp.zeros(ntheta, dtype=jnp.float64)

    m = int((L + 1) / 2)
    x1 = 0.5 * (x2 - x1)
    i = jnp.arange(1, m + 1)
    z = jnp.cos(jnp.pi * (i - 0.25) / (L + 0.5))
    z1 = 2.0 * jnp.ones_like(z)
    pp = jnp.zeros_like(z)

    def optimizer(z, z1, pp):
        def cond(arg):
            z, z1, pp = arg
            return jnp.max(jnp.abs(z - z1)) > 1e-14

        def body(arg):
            z, z1, pp = arg
            p1 = 1.0
            p2 = 0.0
            for j in range(1, L + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = L * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            return z, z1, pp

        return jax.lax.while_loop(cond, body, (z, z1, pp))

    z, z1, pp = optimizer(z, z1, pp)
    weights = weights.at[i - 1].set(2.0 * x1 / ((1.0 - z**2) * pp * pp))
    weights = weights.at[L + 1 - i - 1].set(weights[i - 1])

    return weights * 2 * jnp.pi / (2 * L - 1)


@_partial(_jit, static_argnums=(0))
def quad_weights_dh(L: int) -> jnp.ndarray:
    r"""
    Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_dh`.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * jnp.pi / (2 * L - 1)


@_partial(_jit, static_argnums=(1))
def quad_weight_dh_theta_only(theta: float, L: int) -> float:
    r"""
    Compute DH quadrature weight for :math:`\theta` integration (only), for given
    :math:`\theta`. JAX implementation of :func:`s2fft.quadrature.quad_weights_dh_theta_only`.

    Args:
        theta (float): :math:`\theta` angle for which to compute weight.

        L (int): Harmonic band-limit.

    Returns:
        float: Weight computed for each :math:`\theta`.

    """
    w = 0.0
    for k in range(0, L):
        w += jnp.sin((2 * k + 1) * theta) / (2 * k + 1)

    w *= 2 / L * jnp.sin(theta)

    return w


@_partial(_jit, static_argnums=(0))
def quad_weights_mw(L: int) -> jnp.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mw`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_mw_theta_only(L) * 2 * jnp.pi / (2 * L - 1)


@_partial(_jit, static_argnums=(0))
def quad_weights_mwss(L: int) -> jnp.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mwss`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_mwss_theta_only(L) * 2 * jnp.pi / (2 * L)


@_partial(_jit, static_argnums=(0))
def quad_weights_mwss_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` integration (only).
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mwss_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta`.

    """
    w = jnp.zeros(2 * L, dtype=jnp.complex128)

    for i in range(-(L - 1) + 1, L + 1):
        w = w.at[i + L - 1].set(mw_weights(i - 1))

    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L)
    q = wr[: L + 1]
    q = q.at[1:L].add(wr[-1:L:-1])

    return q


@_partial(_jit, static_argnums=(0))
def quad_weights_mw_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` integration (only).
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mw_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta`.

    """
    w = jnp.zeros(2 * L - 1, dtype=jnp.complex128)
    for i in range(-(L - 1), L):
        w = w.at[i + L - 1].set(mw_weights(i))

    w *= jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1))
    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]
    q = q.at[: L - 1].add(wr[-1 : L - 1 : -1])

    return q


@_partial(_jit, static_argnums=(0))
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
        return 1j * jnp.pi / 2

    elif m == -1:
        return -1j * jnp.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0
