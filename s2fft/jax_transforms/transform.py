from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

from s2fft import samples
import s2fft.healpix_ffts as hp
from s2fft.jax_transforms.otf_recursions import *


def inverse(
    flm: np.ndarray,
    L: int,
    spin: int,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps=None,
) -> np.ndarray:
    if method == "numpy":
        return inverse_numpy(flm, L, spin, nside, sampling, reality, precomps)
    elif method == "jax":
        return inverse_jax(flm, L, spin, nside, sampling, reality, precomps)
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


def inverse_numpy(
    flm: np.ndarray,
    L: int,
    spin: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> np.ndarray:

    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling, nside)
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0

    # Apply harmonic normalisation
    flm = np.einsum(
        "lm,l->lm", flm, np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    )

    # Perform latitudinal wigner-d recursions
    ftm = latitudinal_step(
        flm, thetas, L, spin, nside, sampling, reality, precomps
    )

    # Remove south pole singularity
    if sampling.lower() in ["mw", "mwss"]:
        ftm[-1] = 0
        ftm[-1, L - 1 + spin + m_offset] = np.nansum(
            (-1) ** abs(np.arange(L) - spin) * flm[:, L - 1 + spin]
        )
    # Remove north pole singularity
    if sampling.lower() == "mwss":
        ftm[0] = 0
        ftm[0, L - 1 - spin + m_offset] = jnp.nansum(flm[:, L - 1 - spin])

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp(L, nside, False, reality)
        m_start_ind = L - 1 if reality else 0
        ftm[:, m_start_ind + m_offset :] *= phase_shifts

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    if sampling.lower() == "healpix":
        if reality:
            ftm[:, m_offset : L - 1 + m_offset] = np.flip(
                np.conj(ftm[:, L - 1 + m_offset + 1 :]), axis=-1
            )
        return hp.healpix_ifft(ftm, L, nside, "numpy", reality)
    else:
        if reality:
            return np.fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            return np.fft.ifft(
                np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward"
            )


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def inverse_jax(
    flm: jnp.ndarray,
    L: int,
    spin: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:

    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling, nside)
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0

    # Apply harmonic normalisation
    flm = jnp.einsum(
        "lm,l->lm",
        flm,
        jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)),
        optimize=True,
    )

    # Perform latitudinal wigner-d recursions
    ftm = latitudinal_step_jax(
        flm, thetas, L, spin, nside, sampling, reality, precomps
    )

    # Remove south pole singularity
    if sampling.lower() in ["mw", "mwss"]:
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.nansum((-1) ** abs(jnp.arange(L) - spin) * flm[:, L - 1 + spin])
        )

    # Remove north pole singularity
    if sampling.lower() == "mwss":
        ftm = ftm.at[0].set(0)
        ftm = ftm.at[0, L - 1 - spin + m_offset].set(
            jnp.nansum(flm[:, L - 1 - spin])
        )

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp_jax(L, nside, False, reality)
        m_start_ind = L - 1 if reality else 0
        ftm = ftm.at[:, m_start_ind + m_offset :].multiply(phase_shifts)

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    if sampling.lower() == "healpix":
        if reality:
            ftm = ftm.at[:, m_offset : L - 1 + m_offset].set(
                jnp.flip(jnp.conj(ftm[:, L - 1 + m_offset + 1 :]), axis=-1)
            )
        return hp.healpix_ifft(ftm, L, nside, "jax", reality)
    else:
        if reality:
            return jnp.fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            ftm = jnp.conj(jnp.fft.ifftshift(ftm, axes=1))
            return jnp.conj(jnp.fft.fft(ftm, axis=1, norm="backward"))
