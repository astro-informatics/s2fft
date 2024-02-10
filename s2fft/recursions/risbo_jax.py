import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(1, 2, 3))
def compute_full(dl: jnp.ndarray, beta: float, L: int, el: int) -> jnp.ndarray:
    r"""Compute Wigner-d at argument :math:`\beta` for full plane using
    Risbo recursion (JAX implementation)

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. At present, for :math:`\ell = 0` the recusion is initialised.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\beta`.

        beta (float): Argument :math:`\beta` at which to compute Wigner-d plane.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el` and `beta`, with full plane computed.
    """

    if el == 0:
        dl = dl.at[el + L - 1, el + L - 1].set(1.0)
        return dl
    if el == 1:
        cosb = jnp.cos(beta)
        sinb = jnp.sin(beta)

        coshb = jnp.cos(beta / 2.0)
        sinhb = jnp.sin(beta / 2.0)
        sqrt2 = jnp.sqrt(2.0)

        dl = dl.at[L - 2, L - 2].set(coshb**2)
        dl = dl.at[L - 2, L - 1].set(sinb / sqrt2)
        dl = dl.at[L - 2, L].set(sinhb**2)

        dl = dl.at[L - 1, L - 2].set(-sinb / sqrt2)
        dl = dl.at[L - 1, L - 1].set(cosb)
        dl = dl.at[L - 1, L].set(sinb / sqrt2)

        dl = dl.at[L, L - 2].set(sinhb**2)
        dl = dl.at[L, L - 1].set(-sinb / sqrt2)
        dl = dl.at[L, L].set(coshb**2)
        return dl
    else:
        coshb = -jnp.cos(beta / 2.0)
        sinhb = jnp.sin(beta / 2.0)
        dd = jnp.zeros((2 * el + 2, 2 * el + 2))

        # First pass
        j = 2 * el - 1
        i = jnp.arange(j)
        k = jnp.arange(j)

        sqrt_jmk = jnp.sqrt(j - k)
        sqrt_kp1 = jnp.sqrt(k + 1)
        sqrt_jmi = jnp.sqrt(j - i)
        sqrt_ip1 = jnp.sqrt(i + 1)

        dlj = dl[k - (el - 1) + L - 1][:, i - (el - 1) + L - 1]

        dd = dd.at[:j, :j].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_jmk, optimize=True) * dlj * coshb
        )
        dd = dd.at[:j, 1 : j + 1].add(
            jnp.einsum("i,k->ki", -sqrt_ip1, sqrt_jmk, optimize=True) * dlj * sinhb
        )
        dd = dd.at[1 : j + 1, :j].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_kp1, optimize=True) * dlj * sinhb
        )
        dd = dd.at[1 : j + 1, 1 : j + 1].add(
            jnp.einsum("i,k->ki", sqrt_ip1, sqrt_kp1, optimize=True) * dlj * coshb
        )

        dl = dl.at[-el + L - 1 : el + 1 + L - 1, -el + L - 1 : el + 1 + L - 1].multiply(
            0.0
        )

        j = 2 * el
        i = jnp.arange(j)
        k = jnp.arange(j)

        # Second pass
        sqrt_jmk = jnp.sqrt(j - k)
        sqrt_kp1 = jnp.sqrt(k + 1)
        sqrt_jmi = jnp.sqrt(j - i)
        sqrt_ip1 = jnp.sqrt(i + 1)

        dl = dl.at[-el + L - 1 : el + L - 1, -el + L - 1 : el + L - 1].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_jmk, optimize=True)
            * dd[:j, :j]
            * coshb,
        )
        dl = dl.at[-el + L - 1 : el + L - 1, L - el : L + el].add(
            jnp.einsum("i,k->ki", -sqrt_ip1, sqrt_jmk, optimize=True)
            * dd[:j, :j]
            * sinhb,
        )
        dl = dl.at[L - el : L + el, -el + L - 1 : el + L - 1].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_kp1, optimize=True)
            * dd[:j, :j]
            * sinhb,
        )
        dl = dl.at[L - el : L + el, L - el : L + el].add(
            jnp.einsum("i,k->ki", sqrt_ip1, sqrt_kp1, optimize=True)
            * dd[:j, :j]
            * coshb,
        )
        return dl / ((2 * el) * (2 * el - 1))
