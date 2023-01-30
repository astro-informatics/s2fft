from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

from s2fft import samples
from s2fft.jax_transforms.otf_recursions import *


def inverse(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    method: str = "numpy",
    precomps=None,
) -> np.ndarray:
    if method == "numpy":
        return inverse_numpy(flm, L, spin, sampling, precomps)
    elif method == "jax":
        return inverse_jax(flm, L, spin, sampling, precomps)
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


def inverse_numpy(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    precomps=None,
) -> np.ndarray:

    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # Apply harmonic normalisation
    flm = np.einsum(
        "lm,l->lm", flm, np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    )

    # Perform latitudinal wigner-d recursions
    ftm = latitudinal_step(flm, thetas, L, -spin, sampling, precomps)

    # Remove south pole singularity
    if sampling in ["mw", "mwss"]:
        ftm[-1] = 0
        ftm[-1, L - 1 + spin + m_offset] = np.nansum(
            (-1) ** abs(np.arange(L) - spin) * flm[:, L - 1 + spin]
        )
    # Remove north pole singularity
    if sampling == "mwss":
        ftm[0] = 0
        ftm[0, L - 1 - spin + m_offset] = jnp.nansum(flm[:, L - 1 - spin])

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    return np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


@partial(jit, static_argnums=(1, 2, 3))
def inverse_jax(
    flm: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    precomps=None,
) -> jnp.ndarray:

    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # Apply harmonic normalisation
    flm = jnp.einsum(
        "lm,l->lm",
        flm,
        jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)),
        optimize=True,
    )

    # Perform latitudinal wigner-d recursions
    ftm = latitudinal_step_jax(flm, thetas, L, -spin, sampling, precomps)

    # Remove south pole singularity
    if sampling in ["mw", "mwss"]:
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.nansum((-1) ** abs(jnp.arange(L) - spin) * flm[:, L - 1 + spin])
        )

    # Remove north pole singularity
    if sampling == "mwss":
        ftm = ftm.at[0].set(0)
        ftm = ftm.at[0, L - 1 - spin + m_offset].set(
            jnp.nansum(flm[:, L - 1 - spin])
        )

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    ftm = jnp.conj(jnp.fft.ifftshift(ftm, axes=1))
    return jnp.conj(jnp.fft.fft(ftm, axis=1, norm="backward"))
