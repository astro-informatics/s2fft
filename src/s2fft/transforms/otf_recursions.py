from functools import partial

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit, local_device_count, pmap

from s2fft.recursions.price_mcewen import (
    _polar_terms,
    _recursion_step,
    _scaled_start_values,
)
from s2fft.sampling import s2_samples as samples


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

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Coefficients ftm with indexing :math:`[\theta, m]`.

    """
    mm = -spin
    ntheta = len(beta)
    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    el = np.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = _scaled_start_values(beta, el, mm, np)
    polar = _polar_terms(beta, mm, np)

    for i in range(2):
        if not (reality and i == 0):
            state = tuple(np.zeros((ntheta, L - L_lower)) for _ in range(3))
            for j in range(min(L + abs(mm), 2 * L - 1)):
                dl, state = _recursion_step(i, j, state, start, el, L, mm, polar, np)
                m = j - (L - 1) if i == 0 else L - 1 - j
                ftm[:, L - 1 + m + m_offset] += np.sum(
                    dl * flm[L_lower:, L - 1 + m], axis=-1
                )

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

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

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
    ftm = jnp.zeros(
        (ntheta, samples.ftm_shape(L, sampling, nside)[1]), dtype=jnp.complex128
    )
    el = jnp.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = _scaled_start_values(beta, el, mm, jnp)
    polar = _polar_terms(beta, mm, jnp)
    # Each half recursion runs L steps to reach m = 0 for all el, then |mm| more steps
    # towards the meeting point (a separate loop, as |mm| may be traced)
    n_steps = jnp.minimum(L + jnp.abs(mm), 2 * L - 1)

    for i in range(2):
        if not (reality and i == 0):

            def eval_recursion(ftm, start, polar):
                def pm_recursion_step(j, args):
                    ftm, state = args
                    dl, state = _recursion_step(
                        i, j, state, start, el, L, mm, polar, jnp
                    )
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
                carry = lax.fori_loop(0, L, pm_recursion_step, (ftm, state))
                return lax.fori_loop(L, n_steps, pm_recursion_step, carry)[0]

            if spmd:
                # TODO: Generalise this to optional device counts.
                ndevices = local_device_count()
                opsdevice = int(ntheta / ndevices)

                ftm = pmap(eval_recursion, in_axes=(0, 1, 0))(
                    ftm.reshape(ndevices, opsdevice, ftm.shape[-1]),
                    [p.reshape(2, ndevices, opsdevice, L - L_lower) for p in start],
                    [p.reshape(ndevices, opsdevice) for p in polar],
                ).reshape(ntheta, ftm.shape[-1])

            else:
                ftm = eval_recursion(ftm, start, polar)

    # Remove south pole singularity
    if samples.includes_south_pole(sampling):
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.nansum(
                (-1) ** abs(jnp.arange(L_lower, L) - spin) * flm[L_lower:, L - 1 + spin]
            )
        )

    # Remove north pole singularity
    if samples.includes_north_pole(sampling):
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

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.

    """
    mm = -spin  # switch to match convention
    ntheta = len(beta)  # Number of theta samples
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    el = np.arange(L_lower, L)
    m_offset = samples.m_offset(L, sampling)

    start = _scaled_start_values(beta, el, mm, np)
    polar = _polar_terms(beta, mm, np)

    for i in range(2):
        if not (reality and i == 0):
            state = tuple(np.zeros((ntheta, L - L_lower)) for _ in range(3))
            for j in range(min(L + abs(mm), 2 * L - 1)):
                dl, state = _recursion_step(i, j, state, start, el, L, mm, polar, np)
                m = j - (L - 1) if i == 0 else L - 1 - j
                flm[L_lower:, L - 1 + m] += np.sum(
                    dl * ftm[:, L - 1 + m + m_offset, None], axis=0
                )

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

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

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

    start = _scaled_start_values(beta, el, mm, jnp)
    polar = _polar_terms(beta, mm, jnp)
    # Each half recursion runs L steps to reach m = 0 for all el, then |mm| more steps
    # towards the meeting point (a separate loop, as |mm| may be traced)
    n_steps = jnp.minimum(L + jnp.abs(mm), 2 * L - 1)

    for i in range(2):
        if not (reality and i == 0):

            def eval_recursion(flm, start, el):
                def pm_recursion_step(j, args):
                    flm, state = args
                    dl, state = _recursion_step(
                        i, j, state, start, el, L, mm, polar, jnp
                    )
                    m = j - (L - 1) if i == 0 else L - 1 - j

                    # Sum into transform vector mth component
                    flm = flm.at[:, L - 1 + m].add(
                        jnp.sum(dl * ftm[:, L - 1 + m + m_offset, None], axis=0)
                    )
                    return flm, state

                state = tuple(
                    jnp.zeros((ntheta, el.shape[0]), dtype=jnp.float64)
                    for _ in range(3)
                )
                carry = lax.fori_loop(0, L, pm_recursion_step, (flm, state))
                return lax.fori_loop(L, n_steps, pm_recursion_step, carry)[0]

            if spmd:
                # TODO: Generalise this to optional device counts.
                ndevices = local_device_count()
                opsdevice = int((L - L_lower) / ndevices)

                flm = flm.at[L_lower:].set(
                    pmap(eval_recursion, in_axes=(0, 2, 0))(
                        flm[L_lower:].reshape(ndevices, opsdevice, 2 * L - 1),
                        [p.reshape(2, ntheta, ndevices, opsdevice) for p in start],
                        el.reshape(ndevices, opsdevice),
                    ).reshape(L - L_lower, 2 * L - 1)
                )
            else:
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
