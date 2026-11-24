from functools import partial as _partial

import jax
import jax.numpy as jnp
from jax import jit as _jit

from s2fft.sampling import s2_samples as samples
from s2fft.utils.quadrature import (
    quad_weights_cc as _quad_weights_cc,
)
from s2fft.utils.quadrature import (
    quad_weights_cc_theta_only as _quad_weights_cc_theta_only,
)
from s2fft.utils.quadrature import (
    quad_weights_f2 as _quad_weights_f2,
)
from s2fft.utils.quadrature import (
    quad_weights_f2_theta_only as _quad_weights_f2_theta_only,
)
from s2fft.utils.quadrature import (
    quad_weights_gl_theta_only as _quad_weights_gl_theta_only,
)


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
            {"mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mwss".

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
        return (
            quad_weights_mwss_theta_only(2 * L)
            * 2
            * jnp.pi
            / samples.nphi_equiang(L, "mwss")
        )

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    elif sampling.lower() == "cc":
        return quad_weights_cc(L)

    elif sampling.lower() == "f2":
        return quad_weights_f2(L)

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
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

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

    elif sampling.lower() == "cc":
        return quad_weights_cc(L)

    elif sampling.lower() == "f2":
        return quad_weights_f2(L)

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
def quad_weights_gl_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute GL quadrature weights for :math:`\theta` integration.

    The weights and the GL sample angles :func:`~s2fft.sampling.s2_samples.thetas` come
    from the same computation, at the same nodes.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta`.

    """
    return jnp.asarray(_quad_weights_gl_theta_only(L))


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
    return quad_weights_gl_theta_only(L) * 2 * jnp.pi / samples.nphi_equiang(L, "gl")


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
    return quad_weights_dh_theta_only(L) * 2 * jnp.pi / samples.nphi_equiang(L, "dh")


@_partial(_jit, static_argnums=(0))
def quad_weights_dh_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute DH quadrature weights for :math:`\theta` integration (only).
    JAX implementation of :func:`s2fft.quadrature.quad_weights_dh_theta_only`.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weight computed for each :math:`\theta`.

    """
    thetas = samples.thetas(L, sampling="dh")

    def increment_weights(k, w):
        w += jnp.sin((2 * k + 1) * thetas) / (2 * k + 1)
        return w

    w = jax.lax.fori_loop(0, L, increment_weights, jnp.zeros_like(thetas))
    w *= 2 / L * jnp.sin(thetas)

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
    return quad_weights_mw_theta_only(L) * 2 * jnp.pi / samples.nphi_equiang(L, "mw")


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
    return (
        quad_weights_mwss_theta_only(L) * 2 * jnp.pi / samples.nphi_equiang(L, "mwss")
    )


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
    w = jnp.concatenate((jnp.zeros(1), jax.vmap(mw_weights)(jnp.arange(-(L - 1), L))))
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
    w = jax.vmap(mw_weights)(jnp.arange(-(L - 1), L))
    w *= jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1))
    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]
    q = q.at[: L - 1].add(wr[-1 : L - 1 : -1])

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
    return jax.lax.cond(
        jnp.logical_or(m == 1, m == -1),
        lambda m: jax.lax.cond(
            m == 1, lambda: 1j * jnp.pi / 2, lambda: -1j * jnp.pi / 2
        ),
        lambda m: jax.lax.cond(
            (m % 2) == 0,
            # For complex types here with 0j to ensure nested cond calls
            # are type stable and consistently output complex values
            lambda m: 2 / (1 - m**2) + 0j,
            lambda m: 0j,
            m,
        ),
        m,
    )


@_partial(_jit, static_argnums=(0))
def quad_weights_cc(L: int) -> jnp.ndarray:
    r"""
    Compute Clenshaw-Curtis quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return _quad_weights_cc(L, xp=jnp)


@_partial(_jit, static_argnums=(0))
def quad_weights_cc_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute Clenshaw-Curtis quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta`.

    References:
       Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195–202. https://doi.org/10.1007/s10543-006-0045-4.

    """
    return _quad_weights_cc_theta_only(L, xp=jnp)


@_partial(_jit, static_argnums=(0))
def quad_weights_f2(L: int) -> jnp.ndarray:
    r"""
    Compute Fejér's second rule quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return _quad_weights_f2(L, xp=jnp)


@_partial(_jit, static_argnums=(0))
def quad_weights_f2_theta_only(L: int) -> jnp.ndarray:
    r"""
    Compute Fejér's second rule quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Weights computed for each :math:`\theta`.

    References:
       Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195–202. https://doi.org/10.1007/s10543-006-0045-4.

    """
    return _quad_weights_f2_theta_only(L, xp=jnp)
