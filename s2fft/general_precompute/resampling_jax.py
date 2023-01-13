import numpy as np
import s2fft.samples as samples
from functools import partial

from jax import jit
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(1, 2))
def mw_to_mwss(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)


@partial(jit, static_argnums=(1, 2))
def mw_to_mwss_theta(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    nphi_mw = 2 * L - 1
    ntheta_mw_ext = 2 * L - 1
    ntheta_mwss_ext = 2 * L

    f_mw_ext = periodic_extension(f_mw, L, spin=spin, sampling="mw")

    fmp_mw_ext = jnp.fft.fftshift(
        jnp.fft.fft(f_mw_ext, axis=0, norm="forward"), axes=0
    )

    fmp_mwss_ext = jnp.zeros((ntheta_mwss_ext, nphi_mw), dtype=jnp.complex128)
    for p in range(0, nphi_mw):
        fmp_mw_ext = fmp_mw_ext.at[0:ntheta_mw_ext, p].multiply(
            jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1))
        )
        fmp_mwss_ext = fmp_mwss_ext.at[1 : ntheta_mw_ext + 1, p].set(
            fmp_mw_ext[0:ntheta_mw_ext, p]
        )

    f_mwss_ext = jnp.fft.ifft(
        jnp.fft.ifftshift(fmp_mwss_ext, axes=0), axis=0, norm="forward"
    )

    f_mwss = unextend(f_mwss_ext, L, sampling="mwss")

    return f_mwss


@partial(jit, static_argnums=(1))
def mw_to_mwss_phi(f_mw: jnp.ndarray, L: int) -> jnp.ndarray:
    ntheta = L + 1
    nphi_mw = 2 * L - 1
    nphi_mwss = 2 * L

    ftm_mw = jnp.fft.fftshift(jnp.fft.fft(f_mw, axis=1, norm="forward"), axes=1)

    f_mwss = jnp.zeros((ntheta, nphi_mwss), dtype=jnp.complex128)
    for t in range(0, ntheta):
        f_mwss = f_mwss.at[t, 1 : nphi_mw + 1].set(ftm_mw[t, 0:nphi_mw])

    f_mwss = jnp.fft.ifft(
        jnp.fft.ifftshift(f_mwss, axes=1), axis=1, norm="forward"
    )

    return f_mwss


@partial(jit, static_argnums=(1, 2, 3))
def periodic_extension(
    f: jnp.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> jnp.ndarray:

    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = samples.ntheta_extension(L, sampling)

    f_ext = jnp.zeros((ntheta_ext, nphi), dtype=jnp.complex128)
    f_ext = f_ext.at[0:ntheta, 0:nphi].set(f[0:ntheta, 0:nphi])
    f_ext = jnp.fft.fftshift(
        jnp.fft.fft(f_ext, axis=1, norm="backward"), axes=1
    )

    m_offset = 1 if sampling == "mwss" else 0
    for m in range(-(L - 1), L):

        for t in range(L + m_offset, 2 * L - 1 + m_offset):
            f_ext = f_ext.at[t, m + L - 1 + m_offset].set(
                (-1) ** (m + spin)
                * f_ext[2 * L - 2 - t + 2 * m_offset, m + L - 1 + m_offset]
            )
    return jnp.fft.ifft(
        jnp.fft.ifftshift(f_ext, axes=1), axis=1, norm="backward"
    )


@partial(jit, static_argnums=(1, 2))
def unextend(f_ext: jnp.ndarray, L: int, sampling: str = "mw") -> jnp.ndarray:
    if sampling.lower() == "mw":
        f = f_ext[0:L, :]

    elif sampling.lower() == "mwss":
        f = f_ext[0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )

    return f


@partial(jit, static_argnums=(1, 2))
def upsample_by_two_mwss(f: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext_up = upsample_by_two_mwss_ext(f_ext, L)
    return unextend(f_ext_up, 2 * L, sampling="mwss")


@partial(jit, static_argnums=(1))
def upsample_by_two_mwss_ext(f_ext: jnp.ndarray, L: int) -> jnp.ndarray:

    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.fft.fftshift(jnp.fft.fft(f_ext, axis=0, norm="forward"), axes=0)

    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = jnp.zeros((ntheta_ext_up, nphi), dtype=jnp.complex128)
    for p in range(0, nphi):
        f_ext_up = f_ext_up.at[L : ntheta_ext + L, p].set(
            f_ext[0:ntheta_ext, p]
        )

    return jnp.fft.ifft(
        jnp.fft.ifftshift(f_ext_up, axes=0), axis=0, norm="forward"
    )


@partial(jit, static_argnums=(1, 2))
def periodic_extension_spatial_mwss(
    f: jnp.ndarray, L: int, spin: int = 0
) -> jnp.ndarray:

    ntheta = L + 1
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.zeros((ntheta_ext, nphi), dtype=jnp.complex128)
    f_ext = f_ext.at[0:ntheta, 0:nphi].set(f[0:ntheta, 0:nphi])
    f_ext = f_ext.at[ntheta:, 0 : 2 * L].set(
        (-1) ** spin
        * jnp.fft.fftshift(jnp.flipud(f[1 : ntheta - 1, 0 : 2 * L]), axes=1)
    )

    return f_ext
