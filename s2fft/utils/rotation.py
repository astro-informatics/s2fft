from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from s2fft.recursions.risbo_jax import compute_full


@partial(jit, static_argnums=(1, 2))
def rotate_flms(
    flm: jnp.ndarray,
    L: int,
    rotation: Tuple[float, float, float],
    dl_array: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Rotates an array of spherical harmonic coefficients by angle rotation.

    Args:
        flm (jnp.ndarray): Array of spherical harmonic coefficients.
        L (int): Harmonic band-limit.
        rotation  (Tuple[float, float, float]): Rotation on the sphere (alpha, beta, gamma).
        dl_array (jnp.ndarray, optional): Precomputed array of reduced Wigner d-function
            coefficients, see :func:~`generate_rotate_dls`. Defaults to None.

    Returns:
        jnp.ndarray: Rotated spherical harmonic coefficients with shape [L,2L-1].

    """
    # Split out angles
    alpha = __exp_array(L, rotation[0])
    gamma = __exp_array(L, rotation[2])
    beta = rotation[1]

    # Create empty arrays
    flm_rotated = jnp.zeros_like(flm)

    dl = (
        dl_array
        if dl_array is not None
        else jnp.zeros((2 * L - 1, 2 * L - 1)).astype(jnp.float64)
    )

    # Perform rotation
    for el in range(L):
        if dl_array is None:
            dl = compute_full(dl, beta, L, el)
        n_max = min(el, L - 1)

        m = jnp.arange(-el, el + 1)
        n = jnp.arange(-n_max, n_max + 1)

        flm_rotated = flm_rotated.at[el, L - 1 + m].add(
            jnp.einsum(
                "mn,n->m",
                jnp.einsum(
                    "mn,m->mn",
                    dl[m + L - 1][:, n + L - 1]
                    if dl_array is None
                    else dl[el, m + L - 1][:, n + L - 1],
                    alpha[m + L - 1],
                    optimize=True,
                ),
                gamma[n + L - 1] * flm[el, n + L - 1],
            )
        )
    return flm_rotated


@partial(jit, static_argnums=(0, 1))
def __exp_array(L: int, x: float) -> jnp.ndarray:
    """Private function to generate rotation arrays for alpha/gamma rotations."""
    return jnp.exp(-1j * jnp.arange(-L + 1, L) * x)


@partial(jit, static_argnums=(0, 1))
def generate_rotate_dls(L: int, beta: float) -> jnp.ndarray:
    """
    Function which recursively generates the complete plane of reduced
        Wigner d-function coefficients at a given rotation beta.

    Args:
        L (int): Harmonic band-limit.
        beta (float): Rotation on the sphere.

    Returns:
        jnp.ndarray: Complete array of [L, 2L-1,2L-1] Wigner d-function coefficients
            for a fixed rotation beta.

    """
    dl = jnp.zeros((L, 2 * L - 1, 2 * L - 1)).astype(jnp.float64)
    dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1)).astype(jnp.float64)
    for el in range(L):
        dl_iter = compute_full(dl_iter, beta, L, el)
        dl = dl.at[el].add(dl_iter)
    return dl
