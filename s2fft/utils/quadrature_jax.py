from jax import jit

from functools import partial
from s2fft.sampling import s2_samples as samples
import jax.numpy as jnp


@partial(jit, static_argnums=(0, 1, 2))
def quad_weights_transform(
    L: int, sampling: str = "mwss", nside: int = 0
) -> jnp.ndarray:
    r"""Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration *to use in transform* for various sampling schemes. JAX implementation of
    :func:`~s2fft.quadrature.quad_weights_transform`.

    Quadrature weights to use in transform for MWSS correspond to quadrature weights
    are twice the base resolution, i.e. 2 * L.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mwss", "dh", "healpix}.  Defaults to "mwss".

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

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


@partial(jit, static_argnums=(0, 1, 2))
def quad_weights(L: int = None, sampling: str = "mw", nside: int = None) -> jnp.ndarray:
    r"""Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration for various sampling schemes. JAX implementation of
    :func:`~s2fft.quadrature.quad_weights`.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

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

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


@partial(jit, static_argnums=(0))
def quad_weights_hp(nside: int) -> jnp.ndarray:
    r"""Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi`
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


@partial(jit, static_argnums=(0))
def quad_weights_dh(L: int) -> jnp.ndarray:
    r"""Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_dh`.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * jnp.pi / (2 * L - 1)


@partial(jit, static_argnums=(1))
def quad_weight_dh_theta_only(theta: float, L: int) -> float:
    r"""Compute DH quadrature weight for :math:`\theta` integration (only), for given
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


@partial(jit, static_argnums=(0))
def quad_weights_mw(L: int) -> jnp.ndarray:
    r"""Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mw`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    return quad_weights_mw_theta_only(L) * 2 * jnp.pi / (2 * L - 1)


@partial(jit, static_argnums=(0))
def quad_weights_mwss(L: int) -> jnp.ndarray:
    r"""Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mwss`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    return quad_weights_mwss_theta_only(L) * 2 * jnp.pi / (2 * L)


@partial(jit, static_argnums=(0))
def quad_weights_mwss_theta_only(L: int) -> jnp.ndarray:
    r"""Compute MWSS quadrature weights for :math:`\theta` integration (only).
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mwss_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.
    """
    w = jnp.zeros(2 * L, dtype=jnp.complex128)

    for i in range(-(L - 1) + 1, L + 1):
        w = w.at[i + L - 1].set(mw_weights(i - 1))

    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L)
    q = wr[: L + 1]
    q = q.at[1:L].add(wr[-1:L:-1])

    return q


@partial(jit, static_argnums=(0))
def quad_weights_mw_theta_only(L: int) -> jnp.ndarray:
    r"""Compute MW quadrature weights for :math:`\theta` integration (only).
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mw_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.
    """
    w = jnp.zeros(2 * L - 1, dtype=jnp.complex128)
    for i in range(-(L - 1), L):
        w = w.at[i + L - 1].set(mw_weights(i))

    w *= jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1))
    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]
    q = q.at[: L - 1].add(wr[-1 : L - 1 : -1])

    return q


@partial(jit, static_argnums=(0))
def mw_weights(m: int) -> float:
    r"""Compute MW weights given as a function of index m.

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
