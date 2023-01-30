from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial

from s2fft import samples
from s2fft.wigner.price_mcewen import generate_precomputes


def latitudinal_step(
    flm: np.ndarray,
    beta: np.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> np.ndarray:

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L)
    nel = len(el)  # Number of harmonic modes.
    ftm = np.zeros(samples.ftm_shape(L, sampling), dtype=np.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = np.ones((2, ntheta, nel), dtype=np.float64)

            dl_iter[1, :, lind:] = np.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[i, :, lind:],
            )

            # Sum into transform vector
            ftm[:, sind + m_offset] = np.nansum(
                dl_iter[0, :, lind:]
                * vsign[sind, lind:]
                * np.exp(lrenorm[i, :, lind:])
                * flm[lind:, sind],
                axis=-1,
            )

            # Sum into transform vector
            ftm[:, sind + sgn + m_offset] = np.nansum(
                dl_iter[1, :, lind - 1 :]
                * vsign[sind + sgn, lind - 1 :]
                * np.exp(lrenorm[i, :, lind - 1 :])
                * flm[lind - 1 :, sind + sgn],
                axis=-1,
            )

            dl_entry = np.zeros((ntheta, nel), dtype=np.float64)
            for m in range(2, L):
                index = indices >= L - m - 1
                lamb[i, :, np.arange(nel)] += cs

                dl_entry = np.where(
                    index,
                    np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb[i])
                    - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                    dl_entry,
                )
                dl_entry[:, -(m + 1)] = 1

                # Sum into transform vector
                ftm[:, sind + sgn * m + m_offset] = np.nansum(
                    dl_entry
                    * vsign[sind + sgn * m]
                    * np.exp(lrenorm[i])
                    * flm[:, sind + sgn * m],
                    axis=-1,
                )

                bigi = 1.0 / abs(dl_entry)
                lbig = np.log(abs(dl_entry))

                dl_iter[0] = np.where(index, bigi * dl_iter[1], dl_iter[0])
                dl_iter[1] = np.where(index, bigi * dl_entry, dl_iter[1])
                lrenorm[i] = np.where(index, lrenorm[i] + lbig, lrenorm[i])

    return ftm


@partial(jit, static_argnums=(2, 3, 4, 5))
def latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    el = jnp.arange(L)
    nel = len(el)  # Number of harmonic modes.
    ftm = jnp.zeros(samples.ftm_shape(L, sampling), dtype=jnp.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, nel), dtype=jnp.float64)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[i, :, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector
            ftm = ftm.at[:, sind + m_offset].set(
                jnp.nansum(
                    dl_iter[0, :, lind:]
                    * vsign[sind, lind:]
                    * jnp.exp(lrenorm[i, :, lind:])
                    * flm[lind:, sind],
                    axis=-1,
                )
            )

            # Sum into transform vector
            ftm = ftm.at[:, sind + sgn + m_offset].set(
                jnp.nansum(
                    dl_iter[1, :, lind - 1 :]
                    * vsign[sind + sgn, lind - 1 :]
                    * jnp.exp(lrenorm[i, :, lind - 1 :])
                    * flm[lind - 1 :, sind + sgn],
                    axis=-1,
                )
            )
            dl_entry = jnp.zeros((ntheta, nel), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                ftm, dl_entry, dl_iter, lamb, lrenorm = args

                index = indices >= L - m - 1
                lamb = lamb.at[i, :, jnp.arange(nel)].add(cs)

                dl_entry = jnp.where(
                    index,
                    jnp.einsum(
                        "l,tl->tl",
                        cpi[m - 1],
                        dl_iter[1] * lamb[i],
                        optimize=True,
                    )
                    - jnp.einsum(
                        "l,tl->tl", cp2[m - 1], dl_iter[0], optimize=True
                    ),
                    dl_entry,
                )
                dl_entry = dl_entry.at[:, -(m + 1)].set(1)

                # Sum into transform vector
                ftm = ftm.at[:, sind + sgn * m + m_offset].set(
                    jnp.nansum(
                        dl_entry
                        * vsign[sind + sgn * m]
                        * jnp.exp(lrenorm[i])
                        * flm[:, sind + sgn * m],
                        axis=-1,
                    )
                )

                bigi = 1.0 / abs(dl_entry)
                lbig = jnp.log(abs(dl_entry))

                dl_iter = dl_iter.at[0].set(
                    jnp.where(index, bigi * dl_iter[1], dl_iter[0])
                )
                dl_iter = dl_iter.at[1].set(
                    jnp.where(index, bigi * dl_entry, dl_iter[1])
                )
                lrenorm = lrenorm.at[i].set(
                    jnp.where(index, lrenorm[i] + lbig, lrenorm[i])
                )
                return ftm, dl_entry, dl_iter, lamb, lrenorm

            ftm, dl_entry, dl_iter, lamb, lrenorm = lax.fori_loop(
                2, L, pm_recursion_step, (ftm, dl_entry, dl_iter, lamb, lrenorm)
            )
    return ftm
