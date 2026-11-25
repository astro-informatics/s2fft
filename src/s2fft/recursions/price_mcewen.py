from functools import partial

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit

from s2fft.recursions.common import (
    fori_half_recursion,
    polar_terms,
    recursion_step,
    scaled_start_values,
    start_values,
    static_extension,
)
from s2fft.sampling import s2_samples as samples
from s2fft.utils.deprecation import warn_precomps_deprecated

# Below this polar angle the slices are taken to first order in beta, with an error of
# at most (el beta)^2 / 2: the recursion loses accuracy for beta below about 1e-150,
# where its three-term coefficients approach the largest floats.
_NORTH_POLE_ANGLE = 1e-100


def _set_pole_values(dl, beta, L, mm, xp):
    r"""
    The slices ``dl``, of shape (2L-1, ntheta, L), with their values at the poles set
    in closed form, to first order in the distance from the pole, for
    :math:`\ell \geq |m^\prime|`. That is
    :math:`d^\ell_{m, m^\prime}(\beta) = \delta_{m, m^\prime}
    + \beta\,\partial_\beta d^\ell_{m, m^\prime}(0)` for :math:`|\beta|` below
    ``_NORTH_POLE_ANGLE`` (and at :math:`\beta = 0`, where the starting values of the
    recursion vanish), with an error of at most :math:`(\ell\beta)^2/2`, and, from
    :math:`d^\ell_{m, m^\prime}(\beta) = (-1)^{\ell + m} d^\ell_{m, -m^\prime}(\pi - \beta)`,
    the same about :math:`\pi` for :math:`\beta` exactly ``np.pi``, where the value is
    :math:`(-1)^{\ell - m^\prime}\delta_{m, -m^\prime}`. The derivative at the pole,
    :math:`-i\langle \ell m|J_y|\ell m^\prime\rangle`, is
    :math:`-\sqrt{(\ell - m^\prime)(\ell + m^\prime + 1)}/2` for
    :math:`m = m^\prime + 1`, :math:`\sqrt{(\ell + m^\prime)(\ell - m^\prime + 1)}/2`
    for :math:`m = m^\prime - 1` and zero otherwise, so derivatives with respect to
    :math:`\beta` hold at the poles too. The recursion must have been run at a safe
    angle instead of these :math:`\beta`, so that its values and derivatives, which
    are discarded, stay finite.
    """
    el = xp.arange(L)
    m = xp.arange(-L + 1, L)[:, None]
    valid = el >= xp.abs(mm)

    def first_order(mm):
        # delta_{m, mm} and the derivative of the slice at beta = 0
        delta = xp.where((m == mm) & valid, 1.0, 0.0)
        slope = xp.where(
            (m == mm + 1) & valid,
            -0.5 * xp.sqrt(xp.maximum((el - mm) * (el + mm + 1), 0)),
            xp.where(
                (m == mm - 1) & valid,
                0.5 * xp.sqrt(xp.maximum((el + mm) * (el - mm + 1), 0)),
                0.0,
            ),
        )
        return delta[:, None, :], slope[:, None, :]

    north, south = _pole_masks(beta, xp)
    delta, slope = first_order(mm)
    dl = xp.where(north[None, :, None], delta + beta[None, :, None] * slope, dl)
    delta, slope = first_order(-mm)
    sign = (1.0 - 2.0 * ((el + m) % 2))[:, None, :]
    south_value = sign * (delta + (np.pi - beta)[None, :, None] * slope)
    return xp.where(south[None, :, None], south_value, dl)


def _pole_masks(beta, xp):
    """
    The polar angles taken in closed form by :func:`_set_pole_values`: those below
    ``_NORTH_POLE_ANGLE`` in magnitude, and those exactly ``np.pi``.
    """
    return xp.abs(beta) < _NORTH_POLE_ANGLE, beta == np.pi


def _log_start_values(beta, el, mm, xp):
    r"""
    Natural logarithms of the starting values from :func:`~s2fft.recursions.common.start_values`, zero where
    the starting value vanishes, in the layout of the precomputed ``lrenorm`` arrays.
    """
    mantissa, exponent = start_values(beta, el, mm, xp)
    positive = mantissa > 0
    log_mantissa = xp.log(xp.where(positive, mantissa, 1.0))
    return xp.where(positive, log_mantissa + exponent * np.log(2), 0.0)


def generate_precomputes(
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    L_lower: int = 0,
) -> list[np.ndarray]:
    r"""
    Compute recursion coefficients with :math:`\mathcal{O}(n_\theta L + L^2)` memory
    overhead, which is :math:`\mathcal{O}(L^2)` for the usual sampling schemes, as for
    the JAX implementation :py:func:`generate_precomputes_jax`.

    The recursions in s2fft no longer use these arrays: they evaluate their starting
    values and coefficients on the fly (see :func:`compute_all_slices_jax`), so any
    precomputes passed to the transforms are ignored. This function is deprecated, and
    calling it warns: it will be removed in a future release. ``lrenorm`` holds the
    logarithms of the starting values, now evaluated directly rather than by a
    running sum.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[np.ndarray]: List of precomputed coefficient arrays.

    Note:
        TODO: this function should be optimised.

    """
    warn_precomps_deprecated("generate_precomputes")
    return _generate_precomputes(L, spin, sampling, nside, forward, L_lower)


def _generate_precomputes(L, spin, sampling, nside, forward, L_lower):
    """Implementation of :func:`generate_precomputes`, without its warning."""
    mm = -spin
    L0 = L_lower
    # Correct for mw to mwss conversion
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        beta = samples.thetas(2 * L, "mwss")
    else:
        beta = samples.thetas(L, sampling, nside)

    if forward and samples.includes_both_poles(sampling):
        beta = beta[1:-1]

    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L0, L)

    # Vectors with indexing -L < m < L adopted throughout
    cpi = np.zeros((L + 1, L - L0), dtype=np.float64)
    cp2 = np.zeros((L + 1, L - L0), dtype=np.float64)

    # Initialising coefficients cp(m)= cplus(l-m). They are infinite or NaN where
    # m > 2 el, without numpy's warnings for those entries.
    with np.errstate(divide="ignore", invalid="ignore"):
        cpi[0] = 2.0 / np.sqrt(2 * el)
        for m in range(2, L + 1):
            cpi[m - 1] = 2.0 / np.sqrt(m * (2 * el + 1 - m))
            cp2[m - 1] = cpi[m - 1] / cpi[m - 2]

    for k in range(L0, L):
        cpi[:, k - L0] = np.roll(cpi[:, k - L0], (L - k - 1), axis=-1)
        cp2[:, k - L0] = np.roll(cp2[:, k - L0], (L - k - 1), axis=-1)
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    # Perform precomputations (these can be done offline)
    msign = np.hstack(((-1) ** (abs(np.arange(L - 1))), np.ones(L)))
    lsign = (-1) ** abs(mm + el)
    vsign = np.einsum("m,l->ml", msign, lsign)
    vsign[: L - 1] *= (-1) ** abs(mm + 1 + L)

    # Logarithms of the starting values of the two half recursions, evaluated
    # directly rather than by a running sum whose rounding errors grow like
    # el^2 log(1/beta) near the poles.
    lrenorm = _log_start_values(beta, el, mm, np)

    indices = np.repeat(np.expand_dims(np.arange(L0, L), 0), ntheta, axis=0)
    return [lrenorm, vsign, cpi, cp2, indices]


def generate_precomputes_jax(
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    L_lower: int = 0,
    betas: jnp.ndarray = None,
) -> list[jnp.ndarray]:
    r"""
    Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    JAX implementation of :func:`~generate_precomputes`.

    The recursions in s2fft no longer use these arrays: they evaluate their starting
    values and coefficients on the fly (see :func:`compute_all_slices_jax`), so any
    precomputes passed to the transforms are ignored. This function is deprecated, and
    calling it warns: it will be removed in a future release. ``lrenorm`` holds the
    logarithms of the starting values, now evaluated directly rather than by a
    running sum.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        beta (jnp.ndarray): Array of polar angles in radians.

    Returns:
        List[jnp.ndarray]: List of precomputed coefficient arrays.

    """
    warn_precomps_deprecated("generate_precomputes_jax")
    return _generate_precomputes_jax(L, spin, sampling, nside, forward, L_lower, betas)


@partial(jit, static_argnums=(0, 2, 3, 4, 5))
def _generate_precomputes_jax(L, spin, sampling, nside, forward, L_lower, betas):
    """Implementation of :func:`generate_precomputes_jax`, without its warning."""
    mm = -spin
    L0 = L_lower
    # Correct for mw to mwss conversion
    if betas is None:
        if forward and sampling.lower() in ["mw", "mwss"]:
            sampling = "mwss"
            beta = samples.thetas(2 * L, "mwss")
        else:
            beta = samples.thetas(L, sampling, nside)

        if forward and samples.includes_both_poles(sampling):
            beta = beta[1:-1]
    else:
        beta = betas

    ntheta = len(beta)  # Number of theta samples
    el = jnp.arange(L0, L)

    # Vectors with indexing -L < m < L adopted throughout
    cpi = jnp.zeros((L + 1, L - L0), dtype=jnp.float64)
    cp2 = jnp.zeros((L + 1, L - L0), dtype=jnp.float64)

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi = cpi.at[0].add(2.0 / jnp.sqrt(2 * el))

    def cpi_cp2_loop(m, args):
        cpi, cp2 = args
        cpi = cpi.at[m - 1].add(2.0 / jnp.sqrt(m * (2 * el + 1 - m)))
        cp2 = cp2.at[m - 1].add(cpi[m - 1] / cpi[m - 2])
        return cpi, cp2

    cpi, cp2 = lax.fori_loop(2, L + 1, cpi_cp2_loop, (cpi, cp2))

    def cpi_cp2_roll_loop(m, args):
        cpi, cp2 = args
        cpi = cpi.at[:, m - L0].set(jnp.roll(cpi[:, m - L0], (L - m - 1), axis=-1))
        cp2 = cp2.at[:, m - L0].set(jnp.roll(cp2[:, m - L0], (L - m - 1), axis=-1))
        return cpi, cp2

    cpi, cp2 = lax.fori_loop(L0, L, cpi_cp2_roll_loop, (cpi, cp2))

    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    # Perform precomputations (these can be done offline)
    msign = jnp.hstack(((-1) ** (abs(jnp.arange(L - 1))), jnp.ones(L)))
    lsign = (-1) ** abs(mm + el)
    vsign = jnp.einsum("m,l->ml", msign, lsign, optimize=True)
    vsign = vsign.at[: L - 1].multiply((-1) ** abs(mm + 1 + L))

    # Logarithms of the starting values of the two half recursions, evaluated
    # directly rather than by a running sum whose rounding errors grow like
    # el^2 log(1/beta) near the poles.
    lrenorm = _log_start_values(beta, el, mm, jnp)

    indices = jnp.repeat(jnp.expand_dims(jnp.arange(L0, L), 0), ntheta, axis=0)

    # Remove redundant nans:
    # - in forward pass these are not accessed, so are irrelevant.
    # - in backward pass the adjoint computation otherwise accumulates these
    #   nans into grads if not explicitly set to zero.
    lrenorm = jnp.nan_to_num(lrenorm, nan=0.0, posinf=0.0, neginf=0.0)
    cpi = jnp.nan_to_num(cpi, nan=0.0, posinf=0.0, neginf=0.0)
    cp2 = jnp.nan_to_num(cp2, nan=0.0, posinf=0.0, neginf=0.0)

    return [lrenorm, vsign, cpi, cp2, indices]


def generate_precomputes_wigner(
    L: int,
    N: int,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    L_lower: int = 0,
) -> list[list[np.ndarray]]:
    r"""
    Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    This is a wrapped extension of :func:`~generate_precomputes` for the case of
    multiple spins, i.e. the Wigner transform over SO(3).

    The recursions in s2fft no longer use these arrays, so any precomputes passed to
    the Wigner transforms are ignored (see :func:`~generate_precomputes`). This
    function is deprecated, and calling it warns: it will be removed in a future
    release.

    Args:
        L (int): Harmonic band-limit.

        N (int): Azimuthal bandlimit

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[List[np.ndarray]]: The five arrays of :func:`~generate_precomputes` for
        each of the 2N-1 values of n, or N if ``reality`` is True.

    Note:
        TODO: this function should be optimised.

    """
    warn_precomps_deprecated("generate_precomputes_wigner")
    precomps = []
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps.append(_generate_precomputes(L, -n, sampling, nside, forward, L_lower))
    return precomps


def generate_precomputes_wigner_jax(
    L: int,
    N: int,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    L_lower: int = 0,
) -> list[jnp.ndarray]:
    r"""
    Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    This is a wrapped extension of :func:`~generate_precomputes` for the case of
    multiple spins, i.e. the Wigner transform over SO(3). JAX implementation of
    :func:`~generate_precomputes_wigner`.

    The recursions in s2fft no longer use these arrays, so any precomputes passed to
    the Wigner transforms are ignored (see :func:`~generate_precomputes_jax`). This
    function is deprecated, and calling it warns: it will be removed in a future
    release.

    Args:
        L (int): Harmonic band-limit.

        N (int): Azimuthal bandlimit

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[jnp.ndarray]: The five arrays of :func:`~generate_precomputes_jax`, stacked
        over n along a new leading axis: ``lrenorm`` and ``vsign`` for each of the 2N-1
        values of n (N if ``reality`` is True), and ``cpi``, ``cp2`` and ``indices``,
        which do not depend on n, once.

    """
    warn_precomps_deprecated("generate_precomputes_wigner_jax")
    return _generate_precomputes_wigner_jax(
        L, N, sampling, nside, forward, reality, L_lower
    )


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def _generate_precomputes_wigner_jax(L, N, sampling, nside, forward, reality, L_lower):
    """Implementation of :func:`generate_precomputes_wigner_jax`, without its warning."""
    lrenorm = []
    vsign = []
    cpi = []
    cp2 = []
    indices = []
    captured_repeats = False
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps = _generate_precomputes_jax(
            L, -n, sampling, nside, forward, L_lower, None
        )
        lrenorm.append(precomps[0])
        vsign.append(precomps[1])
        if not captured_repeats:
            cpi.append(precomps[2])
            cp2.append(precomps[3])
            indices.append(precomps[4])
            captured_repeats = True

    return [
        jnp.asarray(lrenorm),
        jnp.asarray(vsign),
        jnp.asarray(cpi),
        jnp.asarray(cp2),
        jnp.asarray(indices),
    ]


def compute_all_slices(
    beta: np.ndarray, L: int, spin: int, precomps=None
) -> np.ndarray:
    r"""
    Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix for all sampled polar angles
    :math:`\beta` and all :math:`\ell` using Price & McEwen recursion.

    The Wigner-d slice for all :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`, by two half recursions running inwards from the edges
    :math:`m = \mp\ell`. Each half recursion is only stable while the slice grows in its
    direction, so the halves meet at :math:`m^\star = \mathrm{round}(m^\prime\cos\beta)`,
    the centre of the oscillatory region of the slice, rather than at :math:`m = 0`;
    near the poles the slice peaks at :math:`m = \pm m^\prime` and decays away from it.

    On-the-fly renormalisation by exact powers of two is implemented to avoid
    over/under-flows, with the scale of the iterates kept as an integer index.

    At the poles the slice is returned in closed form, to first order in the distance
    from the pole, so that derivatives with respect to :math:`\beta` hold there too:
    from :math:`d^\ell_{m, m^\prime}(0) = \delta_{m, m^\prime}` for
    :math:`|\beta| < 10^{-100}` (including :math:`\beta = 0`, where the starting values
    of the recursion vanish), within :math:`(\ell\beta)^2/2`, and from
    :math:`d^\ell_{m, m^\prime}(\pi) = (-1)^{\ell - m^\prime}\delta_{m, -m^\prime}`,
    for :math:`\ell \geq |m^\prime|`, at :math:`\beta` exactly ``np.pi``. A ring
    computed as, say, :math:`\pi(2t + 1)/(2L - 1)` may differ from ``np.pi`` in its
    last bit, and then gets the recursion's value, accurate to round-off.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 + m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires
    :math:`\sim 2(L + |m^\prime|)` operations.

    Args:
        beta (np.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

    Returns:
        np.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.

    """
    if precomps is not None:
        warn_precomps_deprecated()
    mm = -spin
    beta = np.asarray(beta)
    ntheta = len(beta)
    el = np.arange(L)
    # The recursion runs at a safe angle where _set_pole_values replaces its values
    north, south = _pole_masks(beta, np)
    safe_beta = np.where(north | south, np.pi / 2, beta)
    start = scaled_start_values(safe_beta, el, mm, np)
    polar = polar_terms(safe_beta, mm, np)

    dl = np.zeros((2 * L - 1, ntheta, L), dtype=np.float64)
    for i in range(2):
        state = tuple(np.zeros((ntheta, L), dtype=np.float64) for _ in range(3))
        for j in range(min(L + abs(mm), 2 * L - 1)):
            value, state = recursion_step(i, j, state, start, el, L, mm, polar, np)
            m = j - (L - 1) if i == 0 else L - 1 - j
            dl[L - 1 + m] += value
    return _set_pole_values(dl, beta, L, mm, np)


def compute_all_slices_jax(
    beta: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    forward: bool = False,
    nside: int = None,
    precomps=None,
) -> jnp.ndarray:
    r"""
    Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix for all sampled polar angles
    :math:`\beta` and all :math:`\ell` using Price & McEwen recursion.

    The Wigner-d slice for all :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`, by two half recursions running inwards from the edges
    :math:`m = \mp\ell`. Each half recursion is only stable while the slice grows in its
    direction, so the halves meet at :math:`m^\star = \mathrm{round}(m^\prime\cos\beta)`,
    the centre of the oscillatory region of the slice, rather than at :math:`m = 0`;
    near the poles the slice peaks at :math:`m = \pm m^\prime` and decays away from it.

    On-the-fly renormalisation by exact powers of two is implemented to avoid
    over/under-flows, with the scale of the iterates kept as an integer index.

    At the poles the slice is returned in closed form, to first order in the distance
    from the pole, so that derivatives with respect to :math:`\beta` hold there too:
    from :math:`d^\ell_{m, m^\prime}(0) = \delta_{m, m^\prime}` for
    :math:`|\beta| < 10^{-100}` (including :math:`\beta = 0`, where the starting values
    of the recursion vanish), within :math:`(\ell\beta)^2/2`, and from
    :math:`d^\ell_{m, m^\prime}(\pi) = (-1)^{\ell - m^\prime}\delta_{m, -m^\prime}`,
    for :math:`\ell \geq |m^\prime|`, at :math:`\beta` exactly ``np.pi``. A ring
    computed as, say, :math:`\pi(2t + 1)/(2L - 1)` may differ from ``np.pi`` in its
    last bit, and then gets the recursion's value, accurate to round-off.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 + m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires
    :math:`\sim 2(L + |m^\prime|)` operations.

    Args:
        beta (jnp.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray], optional): Deprecated and ignored, and passing it
            warns: the recursion evaluates its starting values and coefficients on the
            fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`. It will be removed in a
            future release.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.

    Raises:
        RuntimeError: If JAX's 64-bit floats are disabled (``jax_enable_x64``).

    """
    if precomps is not None:
        warn_precomps_deprecated()
    return _compute_all_slices_jax(beta, L, spin, static_extension(spin, L))


@partial(jit, static_argnums=(1, 3))
def _compute_all_slices_jax(beta, L, spin, extension):
    """Implementation of :func:`compute_all_slices_jax`."""
    mm = -spin
    ntheta = len(beta)
    el = jnp.arange(L)
    # The recursion runs at a safe angle where _set_pole_values replaces its values
    north, south = _pole_masks(beta, jnp)
    safe_beta = jnp.where(north | south, np.pi / 2, beta)
    start = scaled_start_values(safe_beta, el, mm, jnp)
    polar = polar_terms(safe_beta, mm, jnp)

    dl = jnp.zeros((2 * L - 1, ntheta, L), dtype=jnp.float64)
    for i in range(2):

        def pm_recursion_step(j, args):
            dl, state = args
            value, state = recursion_step(i, j, state, start, el, L, mm, polar, jnp)
            m = j - (L - 1) if i == 0 else L - 1 - j
            dl = dl.at[L - 1 + m].add(value)
            return dl, state

        state = tuple(jnp.zeros((ntheta, L), dtype=jnp.float64) for _ in range(3))
        dl, _ = fori_half_recursion(pm_recursion_step, (dl, state), L, mm, extension)
    return _set_pole_values(dl, beta, L, mm, jnp)
