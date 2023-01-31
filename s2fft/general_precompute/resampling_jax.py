from jax import jit, config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft.samples as samples
from functools import partial


@partial(jit, static_argnums=(1))
def mw_to_mwss(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    if f_mw.ndim == 2:
        return jnp.squeeze(
            mw_to_mwss_phi(
                mw_to_mwss_theta(jnp.expand_dims(f_mw, 0), L, spin), L
            )
        )
    else:
        return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)


@partial(jit, static_argnums=(1))
def mw_to_mwss_theta(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:

    f_mw_ext = periodic_extension(f_mw, L, spin=spin, sampling="mw")
    fmp_mwss_ext = jnp.zeros(
        (f_mw_ext.shape[0], 2 * L, 2 * L - 1), dtype=jnp.complex64
    )

    fmp_mwss_ext = fmp_mwss_ext.at[:, 1:, :].set(
        jnp.fft.fftshift(
            jnp.fft.fft(f_mw_ext, axis=-2, norm="forward"), axes=-2
        )
    )

    fmp_mwss_ext = fmp_mwss_ext.at[:, 1:, :].set(
        jnp.einsum(
            "...blp,l->...blp",
            fmp_mwss_ext[:, 1:, :],
            jnp.exp(-1j * jnp.arange(-(L - 1), L) * jnp.pi / (2 * L - 1)),
            optimize=True,
        )
    )

    f_mwss_ext = jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(fmp_mwss_ext), axes=-2),
            axis=-2,
            norm="backward",
        )
    )

    return unextend(f_mwss_ext, L, sampling="mwss")


@partial(jit, static_argnums=(1))
def mw_to_mwss_phi(f_mw: jnp.ndarray, L: int) -> jnp.ndarray:
    f_mwss = jnp.zeros((f_mw.shape[0], L + 1, 2 * L), dtype=jnp.complex64)
    f_mwss = f_mwss.at[:, :, 1:].set(
        jnp.fft.fftshift(jnp.fft.fft(f_mw, axis=-1, norm="forward"), axes=-1)
    )

    return jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(f_mwss), axes=-1),
            axis=-1,
            norm="backward",
        )
    )


@partial(jit, static_argnums=(1, 3))
def periodic_extension(
    f: jnp.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> jnp.ndarray:
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = samples.ntheta_extension(L, sampling)
    m_offset = 1 if sampling == "mwss" else 0

    f_ext = jnp.zeros((f.shape[0], ntheta_ext, nphi), dtype=jnp.complex64)
    f_ext = f_ext.at[:, 0:ntheta, 0:nphi].set(f[:, 0:ntheta, 0:nphi])
    f_ext = jnp.fft.fftshift(
        jnp.fft.fft(f_ext, axis=-1, norm="backward"), axes=-1
    )

    f_ext = f_ext.at[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ].set(
        jnp.flip(
            f_ext[
                :,
                m_offset : L - 1 + m_offset,
                m_offset : 2 * L - 1 + m_offset,
            ],
            axis=-2,
        )
    )
    f_ext = f_ext.at[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ].multiply((-1) ** (jnp.arange(-(L - 1), L)))
    if spin.size > 1:
        f_ext = f_ext.at[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ].set(
            jnp.einsum(
                "nlm,n->nlm",
                f_ext[
                    :,
                    L + m_offset : 2 * L - 1 + m_offset,
                    m_offset : 2 * L - 1 + m_offset,
                ],
                (-1) ** spin,
                optimize=True,
            )
        )
    else:
        f_ext = f_ext.at[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ].multiply((-1) ** spin)

    return (
        jnp.conj(
            jnp.fft.fft(
                jnp.fft.ifftshift(jnp.conj(f_ext), axes=-1),
                axis=-1,
                norm="backward",
            )
        )
        / nphi
    )


@partial(jit, static_argnums=(1, 2))
def unextend(f_ext: jnp.ndarray, L: int, sampling: str = "mw") -> jnp.ndarray:
    if sampling.lower() == "mw":
        return f_ext[:, 0:L, :]

    elif sampling.lower() == "mwss":
        return f_ext[:, 0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )


@partial(jit, static_argnums=(1))
def upsample_by_two_mwss(f: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    if f.ndim == 2:
        f = jnp.expand_dims(f, 0)
    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext = upsample_by_two_mwss_ext(f_ext, L)
    f_ext = unextend(f_ext, 2 * L, sampling="mwss")
    return jnp.squeeze(f_ext)


@partial(jit, static_argnums=(1))
def upsample_by_two_mwss_ext(f_ext: jnp.ndarray, L: int) -> jnp.ndarray:

    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.fft.fftshift(
        jnp.fft.fft(f_ext, axis=-2, norm="forward"), axes=-2
    )

    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = jnp.zeros(
        (f_ext.shape[0], ntheta_ext_up, nphi), dtype=jnp.complex64
    )
    f_ext_up = f_ext_up.at[:, L : ntheta_ext + L, :nphi].set(
        f_ext[:, 0:ntheta_ext, :nphi]
    )
    return jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(f_ext_up), axes=-2),
            axis=-2,
            norm="backward",
        )
    )


@partial(jit, static_argnums=(1))
def periodic_extension_spatial_mwss(
    f: jnp.ndarray, L: int, spin: int = 0
) -> jnp.ndarray:

    ntheta = L + 1
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.zeros((f.shape[0], ntheta_ext, nphi), dtype=jnp.complex64)
    f_ext = f_ext.at[:, 0:ntheta, 0:nphi].set(f[:, 0:ntheta, 0:nphi])
    if spin.size > 1:
        f_ext = f_ext.at[:, ntheta:, 0 : 2 * L].set(
            jnp.einsum(
                "btp,b->btp",
                jnp.fft.fftshift(
                    jnp.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], axis=-2), axes=-1
                ),
                (-1) ** spin,
                optimize=True,
            ),
        )
    else:
        f_ext = f_ext.at[:, ntheta:, 0 : 2 * L].set(
            (-1) ** spin
            * jnp.fft.fftshift(
                jnp.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], axis=-2), axes=-1
            ),
        )
    return f_ext
