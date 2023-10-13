import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, pmap, local_device_count
from functools import partial
from typing import List
from s2fft.sampling import s2_samples as samples
from s2fft.recursions.price_mcewen import (
    generate_precomputes,
    generate_precomputes_jax,
)


def inverse_latitudinal_step(
    flm: np.ndarray,
    beta: np.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
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

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.
    """
    mm = -spin
    ntheta = len(beta)
    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    el = np.arange(L_lower, L)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    omc = 1.0 - c

    # Indexing boundaries
    lims = [0, -1]
    half_slices = [el + mm + 1, el - mm + 1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside, L_lower)
    lrenorm, vsign, cpi, cp2, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1 - L_lower
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = np.ones((2, ntheta, L - L_lower), dtype=np.float64)

            lamb = (
                np.einsum("l,t->tl", el + 1, omc)
                + np.einsum("l,t->tl", 2 - L + el, c)
                - half_slices[i]
            )
            lamb = np.einsum("tl,t->tl", lamb, 1 / s)

            dl_iter[1, :, lind:] = np.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[:, lind:],
            )

            # Sum into transform vector 0th component
            ftm[:, sind + m_offset] = np.nansum(
                dl_iter[0, :, lind:]
                * vsign[sind, lind:]
                * np.exp(lrenorm[i, :, lind:])
                * flm[lind + L_lower :, sind],
                axis=-1,
            )

            # Sum into transform vector 1st component
            ftm[:, sind + sgn + m_offset] = np.nansum(
                dl_iter[1, :, lind - 1 :]
                * vsign[sind + sgn, lind - 1 :]
                * np.exp(lrenorm[i, :, lind - 1 :])
                * flm[lind - 1 + L_lower :, sind + sgn],
                axis=-1,
            )

            dl_entry = np.zeros((ntheta, L), dtype=np.float64)
            for m in range(2, L - 1 + i):
                index = indices >= L - m - 1
                # lamb[i, :, np.arange(L - L_lower)] += cs

                lamb = (
                    np.einsum("l,t->tl", el + 1, omc)
                    + np.einsum("l,t->tl", m - L + el + 1, c)
                    - half_slices[i]
                )
                lamb = np.einsum("tl,t->tl", lamb, 1 / s)

                dl_entry[:, L_lower:] = np.where(
                    index,
                    np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb)
                    - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                    dl_entry[:, L_lower:],
                )
                dl_entry[:, -(m + 1)] = 1

                # Sum into transform vector nth component
                ftm[:, sind + sgn * m + m_offset] = np.nansum(
                    dl_entry[:, L_lower:]
                    * vsign[sind + sgn * m]
                    * np.exp(lrenorm[i])
                    * flm[L_lower:, sind + sgn * m],
                    axis=-1,
                )

                bigi = 1.0 / abs(dl_entry[:, L_lower:])
                lbig = np.log(abs(dl_entry[:, L_lower:]))

                dl_iter[0] = np.where(index, bigi * dl_iter[1], dl_iter[0])
                dl_iter[1] = np.where(index, bigi * dl_entry[:, L_lower:], dl_iter[1])
                lrenorm[i] = np.where(index, lrenorm[i] + lbig, lrenorm[i])

    return ftm


@partial(jit, static_argnums=(2, 4, 5, 6, 8, 9))
def inverse_latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
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

        precomps (List[jnp.ndarray], optional): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices. Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    m_count = 2 * L if sampling.lower() in ["mwss", "healpix"] else 2 * L - 1
    ftm = jnp.zeros((ntheta, m_count), dtype=jnp.complex128)
    el = jnp.arange(L_lower, L)

    # Trigonometric constant adopted throughout
    c = jnp.cos(beta)
    s = jnp.sin(beta)
    omc = 1.0 - c

    # Indexing boundaries
    lims = [0, -1]
    half_slices = [el + mm + 1, el - mm + 1]

    if precomps is None:
        precomps = generate_precomputes_jax(
            L, -mm, sampling, nside, L_lower=L_lower, betas=beta
        )
    lrenorm, vsign, cpi, cp2, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1 - L_lower
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, L - L_lower), dtype=jnp.float64)

            lamb = (
                jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
                + jnp.einsum("l,t->tl", 2 - L + el, c, optimize=True)
                - half_slices[i]
            )
            lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[:, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector 0th component
            ftm = ftm.at[:, sind + m_offset].set(
                jnp.nansum(
                    dl_iter[0, :, lind:]
                    * vsign[sind, lind:]
                    * jnp.exp(lrenorm[i, :, lind:])
                    * flm[lind + L_lower :, sind],
                    axis=-1,
                )
            )

            # Sum into transform vector 1st component
            ftm = ftm.at[:, sind + sgn + m_offset].set(
                jnp.nansum(
                    dl_iter[1, :, lind - 1 :]
                    * vsign[sind + sgn, lind - 1 :]
                    * jnp.exp(lrenorm[i, :, lind - 1 :])
                    * flm[lind - 1 + L_lower :, sind + sgn],
                    axis=-1,
                )
            )
            dl_entry = jnp.zeros((ntheta, L), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s = args
                index = indices >= L - m - 1

                lamb = (
                    jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
                    + jnp.einsum("l,t->tl", m - L + el + 1, c, optimize=True)
                    - half_slices[i]
                )
                lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

                dl_entry = dl_entry.at[:, L_lower:].set(
                    jnp.where(
                        index,
                        jnp.einsum(
                            "l,tl->tl",
                            cpi[m - 1],
                            dl_iter[1] * lamb,
                            optimize=True,
                        )
                        - jnp.einsum("l,tl->tl", cp2[m - 1], dl_iter[0], optimize=True),
                        dl_entry[:, L_lower:],
                    )
                )
                dl_entry = dl_entry.at[:, -(m + 1)].set(1)

                # Sum into transform vector nth component
                ftm = ftm.at[:, sind + sgn * m + m_offset].set(
                    jnp.nansum(
                        dl_entry[:, L_lower:]
                        * vsign[sind + sgn * m]
                        * jnp.exp(lrenorm[i])
                        * flm[L_lower:, sind + sgn * m],
                        axis=-1,
                    )
                )

                bigi = 1.0 / abs(dl_entry[:, L_lower:])
                lbig = jnp.log(abs(dl_entry[:, L_lower:]))

                dl_iter = dl_iter.at[0].set(
                    jnp.where(index, bigi * dl_iter[1], dl_iter[0])
                )
                dl_iter = dl_iter.at[1].set(
                    jnp.where(index, bigi * dl_entry[:, L_lower:], dl_iter[1])
                )
                lrenorm = lrenorm.at[i].set(
                    jnp.where(index, lrenorm[i] + lbig, lrenorm[i])
                )
                return ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s

            if spmd:

                def eval_recursion_step(
                    ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s
                ):
                    (
                        ftm,
                        dl_entry,
                        dl_iter,
                        lrenorm,
                        indices,
                        omc,
                        c,
                        s,
                    ) = lax.fori_loop(
                        2,
                        L - 1 + i,
                        pm_recursion_step,
                        (ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s),
                    )
                    return ftm

                # TODO: Generalise this to optional device counts.
                ndevices = local_device_count()
                opsdevice = int(ntheta / ndevices)

                ftm = pmap(eval_recursion_step, in_axes=(0, 0, 1, 1, 0, 0, 0, 0))(
                    ftm.reshape(ndevices, opsdevice, ftm.shape[-1]),
                    dl_entry.reshape(ndevices, opsdevice, L),
                    dl_iter.reshape(2, ndevices, opsdevice, L - L_lower),
                    lrenorm.reshape(2, ndevices, opsdevice, L - L_lower),
                    indices.reshape(ndevices, opsdevice, L - L_lower),
                    omc.reshape(ndevices, opsdevice),
                    c.reshape(ndevices, opsdevice),
                    s.reshape(ndevices, opsdevice),
                ).reshape(ntheta, ftm.shape[-1])

            else:
                (ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s,) = lax.fori_loop(
                    2,
                    L - 1 + i,
                    pm_recursion_step,
                    (ftm, dl_entry, dl_iter, lrenorm, indices, omc, c, s),
                )

    # Remove south pole singularity
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0
    if sampling.lower() in ["mw", "mwss"]:
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.nansum(
                (-1) ** abs(jnp.arange(L_lower, L) - spin) * flm[L_lower:, L - 1 + spin]
            )
        )

    # Remove north pole singularity
    if sampling.lower() == "mwss":
        ftm = ftm.at[0].set(0)
        ftm = ftm.at[0, L - 1 - spin + m_offset].set(
            jnp.nansum(flm[L_lower:, L - 1 - spin])
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
    precomps: List = None,
    L_lower: int = 0,
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

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    el = np.arange(L_lower, L)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    omc = 1.0 - c

    # Indexing boundaries
    lims = [0, -1]
    half_slices = [el + mm + 1, el - mm + 1]

    if precomps is None:
        precomps = generate_precomputes(L, -mm, sampling, nside, True, L_lower)
    lrenorm, vsign, cpi, cp2, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1 - L_lower
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = np.ones((2, ntheta, L - L_lower), dtype=np.float64)

            lamb = (
                np.einsum("l,t->tl", el + 1, omc)
                + np.einsum("l,t->tl", 2 - L + el, c)
                - half_slices[i]
            )
            lamb = np.einsum("tl,t->tl", lamb, 1 / s)

            dl_iter[1, :, lind:] = np.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[:, lind:],
            )

            # Sum into transform vector 0th component
            flm[lind + L_lower :, sind] = np.nansum(
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
            flm[lind - 1 + L_lower :, sind + sgn] = np.nansum(
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
            for m in range(2, L - 1 + i):
                index = indices >= L - m - 1
                lamb = (
                    np.einsum("l,t->tl", el + 1, omc)
                    + np.einsum("l,t->tl", m - L + el + 1, c)
                    - half_slices[i]
                )
                lamb = np.einsum("tl,t->tl", lamb, 1 / s)

                dl_entry[:, L_lower:] = np.where(
                    index,
                    np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb)
                    - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                    dl_entry[:, L_lower:],
                )
                dl_entry[:, -(m + 1)] = 1

                # Sum into transform vector nth component
                flm[L_lower:, sind + sgn * m] = np.nansum(
                    np.einsum(
                        "tl, t->tl",
                        dl_entry[:, L_lower:]
                        * vsign[sind + sgn * m]
                        * np.exp(lrenorm[i]),
                        ftm[:, sind + sgn * m + m_offset],
                    ),
                    axis=-2,
                )

                bigi = 1.0 / abs(dl_entry[:, L_lower:])
                lbig = np.log(abs(dl_entry[:, L_lower:]))

                dl_iter[0] = np.where(index, bigi * dl_iter[1], dl_iter[0])
                dl_iter[1] = np.where(index, bigi * dl_entry[:, L_lower:], dl_iter[1])
                lrenorm[i] = np.where(index, lrenorm[i] + lbig, lrenorm[i])

    return flm


@partial(jit, static_argnums=(2, 4, 5, 6, 8, 9))
def forward_latitudinal_step_jax(
    ftm_in: jnp.ndarray,
    beta_in: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
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

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices. Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Spherical harmonic coefficients.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    # Avoid pole-singularities for MWSS sampling
    if sampling.lower() == "mwss":
        ftm = ftm_in[1:-1]
        beta = beta_in[1:-1]
    elif sampling.lower() == "mw":
        ftm = ftm_in[:-1]
        beta = beta_in[:-1]
    else:
        ftm = ftm_in
        beta = beta_in

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128)
    el = jnp.arange(L_lower, L)

    # Trigonometric constant adopted throughout
    c = jnp.cos(beta)
    s = jnp.sin(beta)
    omc = 1.0 - c

    # Indexing boundaries
    lims = [0, -1]
    half_slices = [el + mm + 1, el - mm + 1]

    if precomps is None:
        precomps = generate_precomputes_jax(
            L, -mm, sampling, nside, True, L_lower, betas=beta
        )
    lrenorm, vsign, cpi, cp2, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1 - L_lower
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, L - L_lower), dtype=jnp.float64)

            lamb = (
                jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
                + jnp.einsum("l,t->tl", 2 - L + el, c, optimize=True)
                - half_slices[i]
            )
            lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[:, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector 0th component
            flm = flm.at[lind + L_lower :, sind].set(
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
            flm = flm.at[lind - 1 + L_lower :, sind + sgn].set(
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
            dl_entry = jnp.zeros((ntheta, L - L_lower), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                (
                    flm,
                    dl_entry,
                    dl_iter,
                    lrenorm,
                    cpi,
                    cp2,
                    vsign,
                    indices,
                ) = args

                index = indices >= L - m - 1
                lamb = (
                    jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
                    + jnp.einsum("l,t->tl", m - L + el + 1, c, optimize=True)
                    - half_slices[i]
                )
                lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

                dl_entry = jnp.where(
                    index,
                    jnp.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb, optimize=True)
                    - jnp.einsum("l,tl->tl", cp2[m - 1], dl_iter[0], optimize=True),
                    dl_entry,
                )
                dl_entry = jnp.where(indices == L - 1 - m, 1, dl_entry)

                # Sum into transform vector nth component
                flm = flm.at[:, sind + sgn * m].set(
                    jnp.nansum(
                        jnp.einsum(
                            "tl, t->tl",
                            dl_entry * vsign[sind + sgn * m] * jnp.exp(lrenorm[i]),
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
                return (
                    flm,
                    dl_entry,
                    dl_iter,
                    lrenorm,
                    cpi,
                    cp2,
                    vsign,
                    indices,
                )

            if spmd:

                def eval_recursion_step(
                    flm, dl_entry, dl_iter, lrenorm, cpi, cp2, vsign, indices
                ):
                    (
                        flm,
                        dl_entry,
                        dl_iter,
                        lrenorm,
                        cpi,
                        cp2,
                        vsign,
                        indices,
                    ) = lax.fori_loop(
                        2,
                        L - 1 + i,
                        pm_recursion_step,
                        (
                            flm,
                            dl_entry,
                            dl_iter,
                            lrenorm,
                            cpi,
                            cp2,
                            vsign,
                            indices,
                        ),
                    )
                    return flm

                # TODO: Generalise this to optional device counts.
                ndevices = local_device_count()
                opsdevice = int((L - L_lower) / ndevices)

                flm = flm.at[L_lower:].set(
                    pmap(eval_recursion_step, in_axes=(0, 1, 2, 2, 1, 1, 1, 1),)(
                        flm[L_lower:].reshape(ndevices, opsdevice, 2 * L - 1),
                        dl_entry.reshape(ntheta, ndevices, opsdevice),
                        dl_iter.reshape(2, ntheta, ndevices, opsdevice),
                        lrenorm.reshape(2, ntheta, ndevices, opsdevice),
                        cpi.reshape(L + 1, ndevices, opsdevice),
                        cp2.reshape(L + 1, ndevices, opsdevice),
                        vsign.reshape(2 * L - 1, ndevices, opsdevice),
                        indices.reshape(ntheta, ndevices, opsdevice),
                    ).reshape(L - L_lower, 2 * L - 1)
                )

            else:
                flm = flm.at[L_lower:].set(
                    lax.fori_loop(
                        2,
                        L - 1 + i,
                        pm_recursion_step,
                        (
                            flm[L_lower:],
                            dl_entry,
                            dl_iter,
                            lrenorm,
                            cpi,
                            cp2,
                            vsign,
                            indices,
                        ),
                    )[0]
                )

    # Include both pole singularities explicitly
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0
    if sampling.lower() in ["mw", "mwss"]:
        flm = flm.at[L_lower:, L - 1 + spin].add(
            (-1) ** abs(jnp.arange(L_lower, L) - spin)
            * ftm_in[-1, L - 1 + spin + m_offset]
        )

    if sampling.lower() == "mwss":
        flm = flm.at[L_lower:, L - 1 - spin].add(ftm_in[0, L - 1 - spin + m_offset])
    return flm
