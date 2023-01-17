from functools import partial
from s2fft import samples

from jax import jit
import jax.numpy as jnp


@partial(jit, static_argnums=(0, 1, 2))
def quad_weights_transform(
    L: int, sampling: str = "mwss", nside: int = 0
) -> jnp.ndarray:

    if sampling.lower() == "mwss":
        return quad_weights_mwss_theta_only(2 * L) * 2 * jnp.pi / (2 * L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


@partial(jit, static_argnums=(0, 1, 2))
def quad_weights(
    L: int = None, sampling: str = "mw", nside: int = None
) -> jnp.ndarray:

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
    npix = 12 * nside**2
    rings = samples.ntheta(sampling="healpix", nside=nside)
    return jnp.ones(rings, dtype=jnp.float32) * 4 * jnp.pi / npix


@partial(jit, static_argnums=(0))
def quad_weights_dh(L: int) -> jnp.ndarray:
    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * jnp.pi / (2 * L - 1)


@partial(jit, static_argnums=(1))
def quad_weight_dh_theta_only(theta: float, L: int) -> float:
    w = 0.0
    for k in range(0, L):
        w += jnp.sin((2 * k + 1) * theta) / (2 * k + 1)

    w *= 2 / L * jnp.sin(theta)

    return w


@partial(jit, static_argnums=(0))
def quad_weights_mw(L: int) -> jnp.ndarray:
    return quad_weights_mw_theta_only(L) * 2 * jnp.pi / (2 * L - 1)


@partial(jit, static_argnums=(0))
def quad_weights_mwss(L: int) -> jnp.ndarray:
    return quad_weights_mwss_theta_only(L) * 2 * jnp.pi / (2 * L)


@partial(jit, static_argnums=(0))
def quad_weights_mwss_theta_only(L: int) -> jnp.ndarray:
    w = jnp.zeros(2 * L, dtype=jnp.complex64)

    for i in range(-(L - 1) + 1, L + 1):
        w = w.at[i + L - 1].set(mw_weights(i - 1))

    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (2 * L)
    q = wr[: L + 1]
    q = q.at[1:L].add(wr[-1:L:-1])

    return q


@partial(jit, static_argnums=(0))
def quad_weights_mw_theta_only(L: int) -> jnp.ndarray:
    w = jnp.zeros(2 * L - 1, dtype=jnp.complex64)
    for i in range(-(L - 1), L):
        w = w.at[i + L - 1].set(mw_weights(i))

    w *= jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1))
    wr = jnp.real(jnp.fft.fft(jnp.fft.ifftshift(w), norm="backward")) / (
        2 * L - 1
    )
    q = wr[:L]
    q = q.at[: L - 1].add(wr[-1 : L - 1 : -1])

    return q


@partial(jit, static_argnums=(0))
def mw_weights(m: int) -> float:

    if m == 1:
        return 1j * jnp.pi / 2

    elif m == -1:
        return -1j * jnp.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0
