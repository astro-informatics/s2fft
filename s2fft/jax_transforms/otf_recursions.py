from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial
from typing import List
from s2fft import samples
from s2fft.wigner.price_mcewen import generate_precomputes


def inverse_latitudinal_step(
    flm: np.ndarray,
    beta: np.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> np.ndarray:
    r"""Evaluate the wigner-d recursion inverse latitundinal step over :math:`\theta`.
    This approach is a heavily engineerd version of the Price & McEwen recursion found in
    :func:`~s2fft.wigner.price_mcewen`, which has at most of :math:`\mathcal{O}(L^2)`
    memory footprint.

    This latitundinal :math:`\theta` step for scalar fields reduces to the associated
    Legendre transform, however our transform supports arbitrary spin :math:`s < L`. By
    construction the Price & McEwen approach recurses over m solely, hence though one must
    recurse :math:`\sim L` times, all :math:`\theta, \ell` entries can be computed
    simultaneously; facilitating GPU/TPU acceleration.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        beta (np.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        np.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.
    """
    mm = -spin
    ntheta = len(beta)
    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = np.ones((2, ntheta, L), dtype=np.float64)

            dl_iter[1, :, lind:] = np.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[i, :, lind:],
            )

            # Sum into transform vector 0th component
            ftm[:, sind + m_offset] = np.nansum(
                dl_iter[0, :, lind:]
                * vsign[sind, lind:]
                * np.exp(lrenorm[i, :, lind:])
                * flm[lind:, sind],
                axis=-1,
            )

            # Sum into transform vector 1st component
            ftm[:, sind + sgn + m_offset] = np.nansum(
                dl_iter[1, :, lind - 1 :]
                * vsign[sind + sgn, lind - 1 :]
                * np.exp(lrenorm[i, :, lind - 1 :])
                * flm[lind - 1 :, sind + sgn],
                axis=-1,
            )

            dl_entry = np.zeros((ntheta, L), dtype=np.float64)
            for m in range(2, L):
                index = indices >= L - m - 1
                lamb[i, :, np.arange(L)] += cs

                dl_entry = np.where(
                    index,
                    np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb[i])
                    - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                    dl_entry,
                )
                dl_entry[:, -(m + 1)] = 1

                # Sum into transform vector nth component
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


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def inverse_latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:
    r"""Evaluate the wigner-d recursion inverse latitundinal step over :math:`\theta`.
    This approach is a heavily engineerd version of the Price & McEwen recursion found in
    :func:`~s2fft.wigner.price_mcewen`, which has at most of :math:`\mathcal{O}(L^2)`
    memory footprint. This is a JAX implementation of :func:`~inverse_latitudinal_step`.

    This latitundinal :math:`\theta` step for scalar fields reduces to the associated
    Legendre transform, however our transform supports arbitrary spin :math:`s < L`. By
    construction the Price & McEwen approach recurses over m solely, hence though one must
    recurse :math:`\sim L` times, all :math:`\theta, \ell` entries can be computed
    simultaneously; facilitating GPU/TPU acceleration.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        beta (jnp.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        jnp.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.
    """

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    ftm = jnp.zeros(samples.ftm_shape(L, sampling, nside), dtype=jnp.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, L), dtype=jnp.float64)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[i, :, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector 0th component
            ftm = ftm.at[:, sind + m_offset].set(
                jnp.nansum(
                    dl_iter[0, :, lind:]
                    * vsign[sind, lind:]
                    * jnp.exp(lrenorm[i, :, lind:])
                    * flm[lind:, sind],
                    axis=-1,
                )
            )

            # Sum into transform vector 1st component
            ftm = ftm.at[:, sind + sgn + m_offset].set(
                jnp.nansum(
                    dl_iter[1, :, lind - 1 :]
                    * vsign[sind + sgn, lind - 1 :]
                    * jnp.exp(lrenorm[i, :, lind - 1 :])
                    * flm[lind - 1 :, sind + sgn],
                    axis=-1,
                )
            )
            dl_entry = jnp.zeros((ntheta, L), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                ftm, dl_entry, dl_iter, lamb, lrenorm = args

                index = indices >= L - m - 1
                lamb = lamb.at[i, :, jnp.arange(L)].add(cs)

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

                # Sum into transform vector nth component
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


def forward_latitudinal_step(
    ftm: np.ndarray,
    beta: np.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> np.ndarray:
    r"""Evaluate the wigner-d recursion forward latitundinal step over :math:`\theta`.
    This approach is a heavily engineerd version of the Price & McEwen recursion found in
    :func:`~s2fft.wigner.price_mcewen`, which has at most of :math:`\mathcal{O}(L^2)`
    memory footprint.

    This latitundinal :math:`\theta` step for scalar fields reduces to the associated
    Legendre transform, however our transform supports arbitrary spin :math:`s < L`. By
    construction the Price & McEwen approach recurses over m solely, hence though one must
    recurse :math:`\sim L` times, all :math:`\theta, \ell` entries can be computed
    simultaneously; facilitating GPU/TPU acceleration.

    Args:
        ftm (np.ndarray): Intermediate coefficients with indexing :math:`[\theta, m]`.

        beta (np.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside, True)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = np.ones((2, ntheta, L), dtype=np.float64)
            dl_iter[1, :, lind:] = np.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[i, :, lind:],
            )

            # Sum into transform vector 0th component
            flm[lind:, sind] = np.nansum(
                np.einsum(
                    "tl, t->tl",
                    dl_iter[0, :, lind:]
                    * vsign[sind, lind:]
                    * np.exp(lrenorm[i, :, lind:]),
                    ftm[:, sind + m_offset],
                ),
                axis=-2,
            )

            # Sum into transform vector 1st component
            flm[lind - 1 :, sind + sgn] = np.nansum(
                np.einsum(
                    "tl, t->tl",
                    dl_iter[1, :, lind - 1 :]
                    * vsign[sind + sgn, lind - 1 :]
                    * np.exp(lrenorm[i, :, lind - 1 :]),
                    ftm[:, sind + sgn + m_offset],
                ),
                axis=-2,
            )

            dl_entry = np.zeros((ntheta, L), dtype=np.float64)
            for m in range(2, L):
                index = indices >= L - m - 1
                lamb[i, :, np.arange(L)] += cs

                dl_entry = np.where(
                    index,
                    np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb[i])
                    - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                    dl_entry,
                )
                dl_entry[:, -(m + 1)] = 1

                # Sum into transform vector nth component
                flm[:, sind + sgn * m] = np.nansum(
                    np.einsum(
                        "tl, t->tl",
                        dl_entry * vsign[sind + sgn * m] * np.exp(lrenorm[i]),
                        ftm[:, sind + sgn * m + m_offset],
                    ),
                    axis=-2,
                )

                bigi = 1.0 / abs(dl_entry)
                lbig = np.log(abs(dl_entry))

                dl_iter[0] = np.where(index, bigi * dl_iter[1], dl_iter[0])
                dl_iter[1] = np.where(index, bigi * dl_entry, dl_iter[1])
                lrenorm[i] = np.where(index, lrenorm[i] + lbig, lrenorm[i])

    return flm


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def forward_latitudinal_step_jax(
    ftm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:
    r"""Evaluate the wigner-d recursion forward latitundinal step over :math:`\theta`.
    This approach is a heavily engineerd version of the Price & McEwen recursion found in
    :func:`~s2fft.wigner.price_mcewen`, which has at most of :math:`\mathcal{O}(L^2)`
    memory footprint. This is a JAX implementation of :func:`~forward_latitudinal_step`.

    This latitundinal :math:`\theta` step for scalar fields reduces to the associated
    Legendre transform, however our transform supports arbitrary spin :math:`s < L`. By
    construction the Price & McEwen approach recurses over m solely, hence though one must
    recurse :math:`\sim L` times, all :math:`\theta, \ell` entries can be computed
    simultaneously; facilitating GPU/TPU acceleration.

    Args:
        ftm (jnp.ndarray): Intermediate coefficients with indexing :math:`[\theta, m]`.

        beta (jnp.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        jnp.ndarray: Spherical harmonic coefficients.
    """

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside, True)
    lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, L), dtype=jnp.float64)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[i, :, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector 0th component
            flm = flm.at[lind:, sind].set(
                jnp.nansum(
                    jnp.einsum(
                        "tl, t->tl",
                        dl_iter[0, :, lind:]
                        * vsign[sind, lind:]
                        * jnp.exp(lrenorm[i, :, lind:]),
                        ftm[:, sind + m_offset],
                        optimize=True,
                    ),
                    axis=-2,
                )
            )

            # Sum into transform vector 1st component
            flm = flm.at[lind - 1 :, sind + sgn].set(
                jnp.nansum(
                    jnp.einsum(
                        "tl, t->tl",
                        dl_iter[1, :, lind - 1 :]
                        * vsign[sind + sgn, lind - 1 :]
                        * jnp.exp(lrenorm[i, :, lind - 1 :]),
                        ftm[:, sind + sgn + m_offset],
                        optimize=True,
                    ),
                    axis=-2,
                )
            )
            dl_entry = jnp.zeros((ntheta, L), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                flm, dl_entry, dl_iter, lamb, lrenorm = args

                index = indices >= L - m - 1
                lamb = lamb.at[i, :, jnp.arange(L)].add(cs)

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

                # Sum into transform vector nth component
                flm = flm.at[:, sind + sgn * m].set(
                    jnp.nansum(
                        jnp.einsum(
                            "tl, t->tl",
                            dl_entry
                            * vsign[sind + sgn * m]
                            * jnp.exp(lrenorm[i]),
                            ftm[:, sind + sgn * m + m_offset],
                            optimize=True,
                        ),
                        axis=-2,
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
                return flm, dl_entry, dl_iter, lamb, lrenorm

            flm, dl_entry, dl_iter, lamb, lrenorm = lax.fori_loop(
                2, L, pm_recursion_step, (flm, dl_entry, dl_iter, lamb, lrenorm)
            )
    return flm
