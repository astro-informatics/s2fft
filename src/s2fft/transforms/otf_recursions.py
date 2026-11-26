from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from s2fft.recursions.common import (
    fori_half_recursion,
    polar_terms,
    recursion_step,
    scaled_start_values,
    static_extension,
)
from s2fft.sampling import s2_samples as samples
from s2fft.utils.deprecation import (
    warn_precomps_deprecated,
    warn_spmd_deprecated,
)


def inverse_latitudinal_step(
    flm: np.ndarray,
    beta: np.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: list = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""
    Evaluate the wigner-d recursion inverse latitundinal step over :math:`\theta`.
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
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.

    """
    if precomps is not None:
        warn_precomps_deprecated()
    mm = -spin
    ntheta = len(beta)
    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    el = np.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = scaled_start_values(beta, el, mm, np)
    polar = polar_terms(beta, mm, np)

    for i in range(2):
        # For a real signal only the columns m >= 0 of ftm are needed, the others
        # following by conjugate symmetry. Half recursion 1 gives them all at spin 0,
        # but at other spins half recursion 0 gives the rows 0 <= m < m* near one pole.
        if not (reality and i == 0 and mm == 0):
            state = tuple(np.zeros((ntheta, L - L_lower)) for _ in range(3))
            for j in range(min(L + abs(mm), 2 * L - 1)):
                dl, state = recursion_step(i, j, state, start, el, L, mm, polar, np)
                m = j - (L - 1) if i == 0 else L - 1 - j
                ftm[:, L - 1 + m + m_offset] += np.sum(
                    dl * flm[L_lower:, L - 1 + m], axis=-1
                )

    return ftm


def inverse_latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: list = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""
    Evaluate the wigner-d recursion inverse latitundinal step over :math:`\theta`.
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
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

        spmd (bool, optional): Deprecated and ignored, and passing True warns: the
            transform runs on one device. It will be removed in a future release.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.

    """
    if spmd:
        warn_spmd_deprecated()
    if precomps is not None:
        warn_precomps_deprecated()
    return _inverse_latitudinal_step_jax(
        flm,
        beta,
        L,
        spin,
        nside,
        sampling,
        reality,
        False,
        L_lower,
        static_extension(spin, L),
    )


@partial(jit, static_argnums=(2, 4, 5, 6, 7, 8, 9))
def _inverse_latitudinal_step_jax(
    flm, beta, L, spin, nside, sampling, reality, spmd, L_lower, extension
):
    """Implementation of :func:`inverse_latitudinal_step_jax`."""
    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    ftm = jnp.zeros(
        (ntheta, samples.ftm_shape(L, sampling, nside)[1]), dtype=jnp.complex128
    )
    el = jnp.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = scaled_start_values(beta, el, mm, jnp)
    polar = polar_terms(beta, mm, jnp)

    for i in range(2):
        # For a real signal only the columns m >= 0 of ftm are needed, the others
        # following by conjugate symmetry. Half recursion 1 gives them all at spin 0,
        # but at other spins half recursion 0 gives the rows 0 <= m < m* near one pole,
        # so it is only skipped at spin 0.

        def eval_recursion(ftm, start, polar):
            def pm_recursion_step(j, args):
                ftm, state = args
                dl, state = recursion_step(i, j, state, start, el, L, mm, polar, jnp)
                m = j - (L - 1) if i == 0 else L - 1 - j

                # Sum into transform vector mth component
                ftm = ftm.at[:, L - 1 + m + m_offset].add(
                    jnp.sum(dl * flm[L_lower:, L - 1 + m], axis=-1)
                )
                return ftm, state

            state = tuple(
                jnp.zeros((ftm.shape[0], L - L_lower), dtype=jnp.float64)
                for _ in range(3)
            )
            return fori_half_recursion(
                pm_recursion_step,
                (ftm, state),
                L,
                mm,
                extension,
                skip_at_zero_spin=reality and i == 0,
            )[0]

        ftm = eval_recursion(ftm, start, polar)

    # Remove south pole singularity
    if samples.includes_south_pole(sampling):
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.sum(
                (-1) ** abs(jnp.arange(L_lower, L) - spin) * flm[L_lower:, L - 1 + spin]
            )
        )

    # Remove north pole singularity
    if samples.includes_north_pole(sampling):
        ftm = ftm.at[0].set(0)
        ftm = ftm.at[0, L - 1 - spin + m_offset].set(
            jnp.sum(flm[L_lower:, L - 1 - spin])
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
    precomps: list = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""
    Evaluate the wigner-d recursion forward latitundinal step over :math:`\theta`.
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
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.

    """
    if precomps is not None:
        warn_precomps_deprecated()
    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    el = np.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = scaled_start_values(beta, el, mm, np)
    polar = polar_terms(beta, mm, np)

    for i in range(2):
        # For a real signal at spin 0 the coefficients m < 0 follow by conjugate
        # symmetry from those m >= 0, which half recursion 1 gives alone. At other spins
        # they do not, and half recursion 0 gives them and the rows 0 <= m < m* near
        # one pole.
        if not (reality and i == 0 and mm == 0):
            state = tuple(np.zeros((ntheta, L - L_lower)) for _ in range(3))
            for j in range(min(L + abs(mm), 2 * L - 1)):
                dl, state = recursion_step(i, j, state, start, el, L, mm, polar, np)
                m = j - (L - 1) if i == 0 else L - 1 - j
                flm[L_lower:, L - 1 + m] += np.sum(
                    dl * ftm[:, L - 1 + m + m_offset, None], axis=0
                )

    return flm


def forward_latitudinal_step_jax(
    ftm_in: jnp.ndarray,
    beta_in: jnp.ndarray,
    L: int,
    spin: int,
    nside: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps: list = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""
    Evaluate the wigner-d recursion forward latitundinal step over :math:`\theta`.
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
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

        spmd (bool, optional): Deprecated and ignored, and passing True warns: the
            transform runs on one device. It will be removed in a future release.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Spherical harmonic coefficients.

    """
    if spmd:
        warn_spmd_deprecated()
    if precomps is not None:
        warn_precomps_deprecated()
    return _forward_latitudinal_step_jax(
        ftm_in,
        beta_in,
        L,
        spin,
        nside,
        sampling,
        reality,
        False,
        L_lower,
        static_extension(spin, L),
    )


@partial(jit, static_argnums=(2, 4, 5, 6, 7, 8, 9))
def _forward_latitudinal_step_jax(
    ftm_in, beta_in, L, spin, nside, sampling, reality, spmd, L_lower, extension
):
    """Implementation of :func:`forward_latitudinal_step_jax`."""
    # Avoid pole-singularities where relevant
    if samples.includes_both_poles(sampling):
        ftm = ftm_in[1:-1]
        beta = beta_in[1:-1]
    elif samples.includes_south_pole(sampling) and not samples.includes_north_pole(
        sampling
    ):
        ftm = ftm_in[:-1]
        beta = beta_in[:-1]
    else:
        ftm = ftm_in
        beta = beta_in

    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128)
    el = jnp.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = scaled_start_values(beta, el, mm, jnp)
    polar = polar_terms(beta, mm, jnp)

    for i in range(2):
        # For a real signal at spin 0 the coefficients m < 0 follow by conjugate
        # symmetry from those m >= 0, which half recursion 1 gives alone. At other spins
        # they do not, and half recursion 0 gives them and the rows 0 <= m < m* near
        # one pole, so it is only skipped at spin 0.

        def eval_recursion(flm, start, el):
            def pm_recursion_step(j, args):
                flm, state = args
                dl, state = recursion_step(i, j, state, start, el, L, mm, polar, jnp)
                m = j - (L - 1) if i == 0 else L - 1 - j

                # Sum into transform vector mth component
                flm = flm.at[:, L - 1 + m].add(
                    jnp.sum(dl * ftm[:, L - 1 + m + m_offset, None], axis=0)
                )
                return flm, state

            state = tuple(
                jnp.zeros((ntheta, el.shape[0]), dtype=jnp.float64) for _ in range(3)
            )
            return fori_half_recursion(
                pm_recursion_step,
                (flm, state),
                L,
                mm,
                extension,
                skip_at_zero_spin=reality and i == 0,
            )[0]

        flm = flm.at[L_lower:].set(eval_recursion(flm[L_lower:], start, el))

    # Include both pole singularities explicitly
    if samples.includes_south_pole(sampling):
        flm = flm.at[L_lower:, L - 1 + spin].add(
            (-1) ** abs(jnp.arange(L_lower, L) - spin)
            * ftm_in[-1, L - 1 + spin + m_offset]
        )
    if samples.includes_north_pole(sampling):
        flm = flm.at[L_lower:, L - 1 - spin].add(ftm_in[0, L - 1 - spin + m_offset])

    return flm
