import logging
import math
import warnings
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit

from s2fft.sampling import s2_samples as samples

warnings.filterwarnings("ignore")

# Stirling series error log(n!) - (n log(n) - n + log(2 pi n) / 2) for n = 0, ..., 15
# (zero for n = 0, where it is not used).
_STIRLING_ERROR_TABLE = np.array(
    [
        0.0,
        0.081061466795327258,
        0.041340695955409294,
        0.027677925684998339,
        0.020790672103765093,
        0.016644691189821192,
        0.013876128823070748,
        0.01189670994589177,
        0.010411265261972096,
        0.0092554621827127329,
        0.0083305634333628713,
        0.0075736754879518408,
        0.0069428401072095299,
        0.0064089941880042071,
        0.0059513701127588477,
        0.0055547335519628014,
    ]
)


# The iterates of the recursion are kept within [2^-400, 2^400] and their scale as an
# integer index k, the value being d 2^(800 k); rescaling multiplies by exact powers of
# two (libsharp's approach). Since |d| <= 1, only k = 0 and k = -1 give values above
# 2^-1200, and k < -1 contributes zero.
_SCALE_LIMIT = 2.0**400
_SCALE_FACTOR = 2.0**800
_SCALE_BITS = 800


def _stirling_error(n, xp):
    r"""
    Error of Stirling's approximation, :math:`\log n! - (n \log n - n + \log(2 \pi n)/2)`,
    for arrays of non-negative integers held as floats.
    """
    big = n >= 16
    nb = xp.where(big, n, 16.0)
    n2 = nb * nb
    series = (
        1 / 12
        - (
            1 / 360
            - (1 / 1260 - (1 / 1680 - (1 / 1188 - 691 / 360360 / n2) / n2) / n2) / n2
        )
        / n2
    ) / nb
    table = xp.asarray(_STIRLING_ERROR_TABLE)[xp.clip(n, 0, 15).astype(int)]
    return xp.where(big, series, table)


def _log_binomial(a, b, xp):
    r"""
    :math:`\log \binom{a + b}{a}` for arrays of non-negative integers held as floats,
    accurate to a few units of round-off relative to its value.

    The leading terms of the log-factorials cancel analytically (as in Loader's
    algorithm for binomial probabilities), so no large quantities are subtracted.
    """
    zero = (a == 0) | (b == 0)
    a = xp.where(zero, 1.0, a)
    b = xp.where(zero, 1.0, b)
    n = a + b
    log_binomial = (
        a * xp.log1p(b / a)
        + b * xp.log1p(a / b)
        + 0.5 * xp.log(n / (2 * np.pi * a * b))
        + _stirling_error(n, xp)
        - _stirling_error(a, xp)
        - _stirling_error(b, xp)
    )
    return xp.where(zero, 0.0, log_binomial)


def _exp2(e, xp):
    r"""
    Exact :math:`2^e` for an array of integers held as floats, flushed to zero for
    :math:`e < -1022`, built from its bit pattern (``jnp.ldexp`` and ``jnp.exp2`` round
    on GPU).
    """
    biased_exponent = xp.clip(e + 1023, 0, 2046).astype(xp.int64)
    if xp is np:
        return (biased_exponent << 52).view(np.float64)
    return lax.bitcast_convert_type(biased_exponent << 52, jnp.float64)


def _start_values(beta, el, mm, xp):
    r"""
    Starting values of the two half recursions over :math:`m` for all :math:`\beta` and
    :math:`\ell`, as exact base-2 exponents and mantissas.

    The half recursions start at the edges of the slice, from

    .. math::

        |d^\ell_{\mp\ell, m^\prime}(\beta)| = \sqrt{\binom{2\ell}{\ell + |m^\prime|}}
        \left(\frac{\sin\beta}{2}\right)^{\ell - |m^\prime|} S^{|m^\prime|},

    with :math:`S = \sin^2(\beta/2)` or :math:`\cos^2(\beta/2)`. These values
    underflow near the poles, so they are evaluated as :math:`M 2^E` with :math:`E` an
    integer: the exponents of :math:`\sin\beta/2` and :math:`S` are split off exactly
    (``frexp``) and only logarithms of :math:`\mathcal{O}(1)` mantissas are taken,
    giving a relative error of a few :math:`\ell` units of round-off rather than one
    proportional to :math:`\ell \log(1/\beta)`.

    Args:
        beta (np.ndarray | jnp.ndarray): Polar angles in radians, shape (ntheta,).

        el (np.ndarray | jnp.ndarray): Harmonic degrees, shape (nel,).

        mm (int): Harmonic order :math:`m^\prime` at which to slice the matrix.

        xp (module): Array namespace, numpy or jax.numpy.

    Returns:
        Tuple[np.ndarray | jnp.ndarray]: Mantissas and (integer-valued) exponents, each
        of shape (2, ntheta, nel), for the half recursions starting at
        :math:`m = -\ell` (index 0) and :math:`m = \ell` (index 1); zero for
        :math:`\ell < |m^\prime|`.

    """
    el = xp.asarray(el, dtype=xp.float64)
    abs_mm = xp.abs(mm)
    valid = el >= abs_mm
    k = xp.where(valid, el - abs_mm, 0.0)

    sin_half_beta_sq = xp.sin(beta / 2) ** 2
    cos_half_beta_sq = xp.cos(beta / 2) ** 2
    frac_t, exp_t = xp.frexp(xp.sin(beta) / 2)
    log_frac_t = xp.log2(xp.where(frac_t > 0, frac_t, 1.0))
    # log2(sqrt(binom(2 el, el + |mm|))) - el, which is O(log el)
    log_binom = _log_binomial(el + abs_mm, k, xp) / (2 * np.log(2)) - el

    mantissas = []
    exponents = []
    for i in range(2):
        s_is_sin = mm > 0 if i == 0 else mm < 0
        frac_s, exp_s = xp.frexp(xp.where(s_is_sin, sin_half_beta_sq, cos_half_beta_sq))
        log_frac_s = xp.log2(xp.where(frac_s > 0, frac_s, 1.0))
        log_mantissa = (
            k[None, :] * log_frac_t[:, None]
            + abs_mm * log_frac_s[:, None]
            + log_binom[None, :]
        )
        exponent = xp.floor(log_mantissa)
        mantissa = xp.exp2(log_mantissa - exponent)
        exponent = (
            exponent
            + k[None, :] * exp_t[:, None]
            + abs_mm * exp_s[:, None]
            + el[None, :]
        )
        zero = (
            ~valid[None, :]
            | ((frac_t == 0)[:, None] & (k > 0)[None, :])
            | ((frac_s == 0) & (abs_mm > 0))[:, None]
        )
        mantissas.append(xp.where(zero, 0.0, mantissa))
        exponents.append(xp.where(zero, 0.0, exponent))
    return xp.stack(mantissas), xp.stack(exponents)


def _stirling_error_scalar(n):
    """Scalar version of :func:`_stirling_error`."""
    if n < 16:
        return float(_STIRLING_ERROR_TABLE[int(n)])
    n2 = n * n
    return (
        1 / 12
        - (
            1 / 360
            - (1 / 1260 - (1 / 1680 - (1 / 1188 - 691 / 360360 / n2) / n2) / n2) / n2
        )
        / n2
    ) / n


def _start_values_scalar(beta, el, mm):
    r"""
    Scalar version of :func:`_start_values` for a single :math:`\beta` and
    :math:`\ell`, for the scalar Turok & Bucher recursion.

    Returns:
        Tuple: Mantissas and exponents of the starting values of the half recursions
        starting at :math:`m = -\ell` and :math:`m = \ell`, as
        ``((mantissa_0, exponent_0), (mantissa_1, exponent_1))``.

    """
    abs_mm = abs(mm)
    if el < abs_mm:
        return (0.0, 0), (0.0, 0)
    k = el - abs_mm
    a, b = el + abs_mm, k
    if a == 0 or b == 0:
        log_binom = 0.0
    else:
        n = a + b
        log_binom = (
            a * math.log1p(b / a)
            + b * math.log1p(a / b)
            + 0.5 * math.log(n / (2 * math.pi * a * b))
            + _stirling_error_scalar(n)
            - _stirling_error_scalar(a)
            - _stirling_error_scalar(b)
        )
    log_binom = log_binom / (2 * math.log(2)) - el
    frac_t, exp_t = math.frexp(math.sin(beta) / 2)
    starts = []
    for i in range(2):
        s_is_sin = mm > 0 if i == 0 else mm < 0
        s = math.sin(beta / 2) ** 2 if s_is_sin else math.cos(beta / 2) ** 2
        frac_s, exp_s = math.frexp(s)
        if (frac_t == 0 and k > 0) or (frac_s == 0 and abs_mm > 0):
            starts.append((0.0, 0))
            continue
        log_mantissa = (
            (k * math.log2(frac_t) if k > 0 else 0.0)
            + (abs_mm * math.log2(frac_s) if abs_mm > 0 else 0.0)
            + log_binom
        )
        exponent = math.floor(log_mantissa)
        starts.append(
            (
                2.0 ** (log_mantissa - exponent),
                exponent + k * exp_t + abs_mm * exp_s + el,
            )
        )
    return tuple(starts)


def _scaled_start_values(beta, el, mm, xp):
    r"""
    Starting values from :func:`_start_values` as iterates within
    :math:`[2^{-400}, 2^{401})` and scale indices :math:`k`, the value being
    :math:`d\,2^{800 k}`; each of shape (2, ntheta, nel).
    """
    if xp is jnp and not jax.config.read("jax_enable_x64"):
        # Logged rather than warned: this module ignores warnings globally
        logging.getLogger("s2fft").warning(
            "The Wigner-d recursions require double precision for their accuracy near "
            "the poles, and give wrong results in single precision: enable it with "
            'jax.config.update("jax_enable_x64", True).'
        )
    mantissa, exponent = _start_values(beta, el, mm, xp)
    scale = xp.rint(exponent / _SCALE_BITS)
    return mantissa * _exp2(exponent - _SCALE_BITS * scale, xp), scale


def _static_extension(spin, L):
    r"""
    Number of steps that the half recursions run past :math:`m = 0`, as a static
    bound: :math:`|m^\prime|` when the spin is concrete, or its largest value
    :math:`L - 1` when the spin is traced (under ``jit`` or ``vmap``). Steps past the
    meeting point change nothing, and a loop of static length can be differentiated
    in reverse mode.
    """
    try:
        return abs(int(spin))
    except (jax.errors.ConcretizationTypeError, TypeError):
        return L - 1


def _fori_half_recursion(body, carry, L, mm, extension):
    r"""
    Run the steps of a half recursion: :math:`L` steps take every :math:`\ell` to
    :math:`m = 0` and ``extension`` more run towards the meeting point, in one loop of
    static length. With ``extension`` None, the :math:`|m^\prime|` extra steps instead
    run in a second loop with a traced trip count. That saves steps when the spin is
    traced, but cannot be differentiated in reverse mode, so it is only for callers
    with their own derivative rules.
    """
    if extension is None:
        carry = lax.fori_loop(0, L, body, carry)
        return lax.fori_loop(L, jnp.minimum(L + jnp.abs(mm), 2 * L - 1), body, carry)
    return lax.fori_loop(0, min(L + extension, 2 * L - 1), body, carry)


def _recursion_step(i, j, state, start, el, L, mm, polar, xp):
    r"""
    Step :math:`j` of half recursion :math:`i` of the Price & McEwen slice recursion.

    Half recursion 0 runs upwards in :math:`m` from the edge :math:`m = -\ell` and half
    recursion 1 downwards from :math:`m = \ell`. Degree :math:`\ell` starts at step
    :math:`L - 1 - \ell`, so step :math:`j` evaluates the row
    :math:`m = j - (L - 1)` (:math:`i = 0`) or :math:`m = L - 1 - j` (:math:`i = 1`)
    for all :math:`\beta` and :math:`\ell` at once. The iterates are rescaled by exact
    powers of two, with the scale kept as an integer index. Each half recursion
    stops at the meeting point :math:`m^\star` from :func:`_polar_terms`: half
    recursion 0 gives the rows :math:`m < m^\star` and half recursion 1 the rows
    :math:`m \geq m^\star`, so that both only run in the direction in which the
    slice grows.

    Args:
        i (int): Half recursion, 0 or 1.

        j (int): Step.

        state (Tuple): Previous and current iterates and their scale index, each of
            shape (ntheta, nel).

        start (Tuple): Starting values and their scale indices from
            :func:`_scaled_start_values`.

        el (np.ndarray | jnp.ndarray): Harmonic degrees, shape (nel,).

        L (int): Harmonic band-limit.

        mm (int): Harmonic order :math:`m^\prime` at which to slice the matrix.

        polar (Tuple): Functions of :math:`\beta` and the meeting point from
            :func:`_polar_terms`.

        xp (module): Array namespace, numpy or jax.numpy.

    Returns:
        Tuple: :math:`d^\ell_{m, m^\prime}(\beta)` on row :math:`m` with shape
        (ntheta, nel), zero for :math:`|m| > \ell` and where the row belongs to the
        other half recursion, and the updated state.

    """
    d_prev, d_curr, scale = state
    hemisphere, half_angle, sin_beta, m_star = polar
    q = j - (L - 1) + el  # Steps from the edge of the slice for each el
    at_edge = q == 0
    stepping = (q >= 1) & (q <= 2 * el)
    q = xp.asarray(q, dtype=xp.float64)
    el = xp.asarray(el, dtype=xp.float64)

    # Coefficients of the three-term recursion, on the fly
    a_curr = xp.where(stepping, q * (2 * el + 1 - q), 1.0)
    a_prev = xp.where(stepping & (q >= 2), (q - 1) * (2 * el + 2 - q), 0.0)
    cpi = 2.0 / xp.sqrt(a_curr)
    cp2 = xp.sqrt(a_prev / a_curr)

    # lamb = -(m_c cos(beta) + mm_i) / sin(beta), where m_c is the distance from the
    # centre of the previous row. Writing m_c cos(beta) + mm_i as
    # (m_c + mm_i) - 2 m_c sin^2(beta/2) in the northern hemisphere and as
    # -((m_c - mm_i) - 2 m_c cos^2(beta/2)) in the southern avoids the cancellation in
    # cos(beta) near the poles.
    m_c = (el + 1 - q)[None, :]
    mm_i = mm if i == 0 else -mm
    integer_part = m_c + hemisphere[:, None] * mm_i  # Exact
    lamb = integer_part - m_c * half_angle[:, None]
    lamb = -hemisphere[:, None] * lamb / sin_beta[:, None]

    # Degree L - 1 - j starts at this step (degrees el are consecutive): read only its
    # column of the starting values.
    edge = xp.clip((L - 1 - j) - el[0], 0, el.shape[0] - 1).astype(int)
    d_next = cpi * lamb * d_curr - cp2 * d_prev
    d_next = xp.where(at_edge, xp.take(start[0][i], edge, axis=1)[:, None], d_next)
    scale = xp.where(at_edge, xp.take(start[1][i], edge, axis=1)[:, None], scale)

    # Past the meeting point the slice decays in the direction of this half recursion,
    # which would amplify round-off, so it stops there (and does not start if its edge
    # lies beyond it).
    m = j - (L - 1) if i == 0 else L - 1 - j
    owned = m < m_star if i == 0 else m >= m_star
    active = (at_edge | stepping) & owned[:, None]

    sign = 1 - 2 * (((m if i == 0 else mm) + el) % 2)
    unscale = xp.where(scale == 0, 1.0, xp.where(scale == -1, 1 / _SCALE_FACTOR, 0.0))
    value = xp.where(active, sign * d_next * unscale, 0.0)

    # Rescale both iterates by an exact power of two when they leave [2^-400, 2^400]
    largest = xp.maximum(xp.abs(d_curr), xp.abs(d_next))
    up = largest > _SCALE_LIMIT
    down = (largest < 1 / _SCALE_LIMIT) & (largest > 0)
    factor = xp.where(up, 1 / _SCALE_FACTOR, xp.where(down, _SCALE_FACTOR, 1.0))
    d_prev = xp.where(active, d_curr * factor, d_prev)
    d_curr = xp.where(active, d_next * factor, d_curr)
    scale = xp.where(active, scale + up - down, scale)
    return value, (d_prev, d_curr, scale)


def _polar_terms(beta, mm, xp):
    r"""
    Functions of :math:`\beta` adopted by the recursion: the hemisphere (:math:`\pm 1`),
    :math:`2\sin^2(\beta/2)` in the north or :math:`2\cos^2(\beta/2)` in the south,
    :math:`\sin\beta` (one at the poles), and the meeting point of the two half
    recursions, :math:`m^\star = \mathrm{round}(m^\prime\cos\beta)`, the centre of the
    oscillatory region of the slice.
    """
    cos_beta = xp.cos(beta)
    sin_beta = xp.sin(beta)
    north = cos_beta >= 0
    return (
        xp.where(north, 1.0, -1.0),
        2 * xp.where(north, xp.sin(beta / 2), xp.cos(beta / 2)) ** 2,
        xp.where(sin_beta == 0, 1.0, sin_beta),
        xp.rint(mm * cos_beta),
    )


def _log_start_values(beta, el, mm, xp):
    r"""
    Natural logarithms of the starting values from :func:`_start_values`, zero where
    the starting value vanishes, in the layout of the precomputed ``lrenorm`` arrays.
    """
    mantissa, exponent = _start_values(beta, el, mm, xp)
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
    precomputes passed to the transforms are ignored. This function is kept for
    backward compatibility; ``lrenorm`` holds the logarithms of the starting values,
    now evaluated directly rather than by a running sum.

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

    # Initialising coefficients cp(m)= cplus(l-m).
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


@partial(jit, static_argnums=(0, 2, 3, 4, 5))
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
    precomputes passed to the transforms are ignored. This function is kept for
    backward compatibility; ``lrenorm`` holds the logarithms of the starting values,
    now evaluated directly rather than by a running sum.

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
    function is kept for backward compatibility.

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
    precomps = []
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps.append(generate_precomputes(L, -n, sampling, nside, forward, L_lower))
    return precomps


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
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
    function is kept for backward compatibility.

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
    lrenorm = []
    vsign = []
    cpi = []
    cp2 = []
    indices = []
    captured_repeats = False
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps = generate_precomputes_jax(L, -n, sampling, nside, forward, L_lower)
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

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 + m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires
    :math:`\sim 2(L + |m^\prime|)` operations.

    Args:
        beta (np.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

    Returns:
        np.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.

    """
    mm = -spin
    beta = np.asarray(beta)
    ntheta = len(beta)
    el = np.arange(L)
    start = _scaled_start_values(beta, el, mm, np)
    polar = _polar_terms(beta, mm, np)

    dl = np.zeros((2 * L - 1, ntheta, L), dtype=np.float64)
    for i in range(2):
        state = tuple(np.zeros((ntheta, L), dtype=np.float64) for _ in range(3))
        for j in range(min(L + abs(mm), 2 * L - 1)):
            value, state = _recursion_step(i, j, state, start, el, L, mm, polar, np)
            m = j - (L - 1) if i == 0 else L - 1 - j
            dl[L - 1 + m] += value
    return dl


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

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 + m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires
    :math:`\sim 2(L + |m^\prime|)` operations.

    Args:
        beta (jnp.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray], optional): Not used, kept for backward
            compatibility: the recursion evaluates its starting values and coefficients
            on the fly, at a cost of :math:`\mathcal{O}(n_{\theta} L)`.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.

    """
    return _compute_all_slices_jax(beta, L, spin, _static_extension(spin, L))


@partial(jit, static_argnums=(1, 3))
def _compute_all_slices_jax(beta, L, spin, extension):
    """Implementation of :func:`compute_all_slices_jax`."""
    mm = -spin
    ntheta = len(beta)
    el = jnp.arange(L)
    start = _scaled_start_values(beta, el, mm, jnp)
    polar = _polar_terms(beta, mm, jnp)

    dl = jnp.zeros((2 * L - 1, ntheta, L), dtype=jnp.float64)
    for i in range(2):

        def pm_recursion_step(j, args):
            dl, state = args
            value, state = _recursion_step(i, j, state, start, el, L, mm, polar, jnp)
            m = j - (L - 1) if i == 0 else L - 1 - j
            dl = dl.at[L - 1 + m].add(value)
            return dl, state

        state = tuple(jnp.zeros((ntheta, L), dtype=jnp.float64) for _ in range(3))
        dl, _ = _fori_half_recursion(pm_recursion_step, (dl, state), L, mm, extension)
    return dl
