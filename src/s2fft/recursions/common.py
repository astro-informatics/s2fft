"""
Helpers shared by the Price-McEwen and Turok-Bucher recursions and by the latitudinal
steps of the transforms: the starting values of the half recursions, their steps and
the loop that runs them, with the iterates rescaled by exact powers of two.
"""

import math

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

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
SCALE_LIMIT = 2.0**400
SCALE_FACTOR = 2.0**800
SCALE_BITS = 800


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


def start_values(beta, el, mm, xp):
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

    Raises:
        RuntimeError: For JAX arrays when 64-bit floats are disabled: the iterates of
            the recursion lie far outside the range of single precision, so it would
            return zeros. The error is raised before any computation, while tracing (or,
            through the transforms, while their primitive is lowered), so no
            single-precision trace is kept for later calls with 64-bit floats.

    """
    if xp is jnp and not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "The Wigner-d recursions of s2fft need 64-bit floats in JAX, and give "
            "wrong results without them. Enable them with "
            'jax.config.update("jax_enable_x64", True) at the start of the program.'
        )
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


def start_values_scalar(beta, el, mm):
    r"""
    Scalar version of :func:`start_values` for a single :math:`\beta` and
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


def scaled_start_values(beta, el, mm, xp):
    r"""
    Starting values from :func:`start_values` as iterates within
    :math:`[2^{-400}, 2^{401})` and scale indices :math:`k`, the value being
    :math:`d\,2^{800 k}`; each of shape (2, ntheta, nel).
    """
    mantissa, exponent = start_values(beta, el, mm, xp)
    scale = xp.rint(exponent / SCALE_BITS)
    # Exact powers of two 2^(exponent - 800 scale), flushed to zero below 2^-1022, built
    # from their bit patterns: jnp.ldexp and jnp.exp2 are not exact powers of two in
    # general (they round on GPU)
    biased_exponent = xp.clip(exponent - SCALE_BITS * scale + 1023, 0, 2046)
    bits = biased_exponent.astype(xp.int64) << 52
    if xp is np:
        power = bits.view(np.float64)
    else:
        power = lax.bitcast_convert_type(bits, jnp.float64)
    return mantissa * power, scale


def static_extension(spin, L):
    r"""
    Number of steps that the half recursions run past :math:`m = 0`, as a static
    bound: :math:`|m^\prime|` when the spin is known, or its largest value
    :math:`L - 1` when it is a JAX tracer (under ``jit`` or ``vmap``). Steps past the
    meeting point change nothing, and a loop of static length can be differentiated
    in reverse mode. A traced spin is supported because s2fft's Wigner transforms
    vmap the spherical ones over it. The bound has to be taken before ``jit`` traces
    the spin, so the JAX recursions and latitudinal steps are thin wrappers that pass
    it to their jitted implementations as a static argument.
    """
    if isinstance(spin, jax.core.Tracer):
        return L - 1
    return abs(int(spin))


def fori_half_recursion(body, carry, L, mm, extension, skip_at_zero_spin=False):
    r"""
    Run the steps of a half recursion: :math:`L` steps take every :math:`\ell` to
    :math:`m = 0` and ``extension`` more run towards the meeting point, in one loop of
    static length. With ``extension`` None, the :math:`|m^\prime|` extra steps instead
    run in a second loop with a traced trip count. That saves steps when the spin is
    traced, but cannot be differentiated in reverse mode, so it is only for callers
    with their own derivative rules.

    With ``skip_at_zero_spin``, nothing runs when :math:`m^\prime = 0`, as for half
    recursion 0 of a real signal at spin 0. Since ``extension`` bounds
    :math:`|m^\prime|`, the spin is zero when it is, so that test is static; with
    ``extension`` None it is a ``lax.cond`` on the traced spin. A traced spin with a
    static ``extension`` (:math:`L - 1`) always runs, also at spin 0: that only costs
    time, and fills the rows :math:`m < 0` that would otherwise stay zero.
    """
    if extension is None:

        def run(carry):
            carry = lax.fori_loop(0, L, body, carry)
            return lax.fori_loop(
                L, jnp.minimum(L + jnp.abs(mm), 2 * L - 1), body, carry
            )

        if skip_at_zero_spin:
            return lax.cond(mm == 0, lambda carry: carry, run, carry)
        return run(carry)
    if skip_at_zero_spin and extension == 0:
        return carry
    return lax.fori_loop(0, min(L + extension, 2 * L - 1), body, carry)


def recursion_step(i, j, state, start, el, L, mm, polar, xp):
    r"""
    Step :math:`j` of half recursion :math:`i` of the Price & McEwen slice recursion.

    Half recursion 0 runs upwards in :math:`m` from the edge :math:`m = -\ell` and half
    recursion 1 downwards from :math:`m = \ell`. Degree :math:`\ell` starts at step
    :math:`L - 1 - \ell`, so step :math:`j` evaluates the row
    :math:`m = j - (L - 1)` (:math:`i = 0`) or :math:`m = L - 1 - j` (:math:`i = 1`)
    for all :math:`\beta` and :math:`\ell` at once. The iterates are rescaled by exact
    powers of two, with the scale kept as an integer index. Each half recursion
    stops at the meeting point :math:`m^\star` from :func:`polar_terms`: half
    recursion 0 gives the rows :math:`m < m^\star` and half recursion 1 the rows
    :math:`m \geq m^\star`, so that both only run in the direction in which the
    slice grows.

    Args:
        i (int): Half recursion, 0 or 1.

        j (int): Step.

        state (Tuple): Previous and current iterates and their scale index, each of
            shape (ntheta, nel).

        start (Tuple): Starting values and their scale indices from
            :func:`scaled_start_values`.

        el (np.ndarray | jnp.ndarray): Harmonic degrees, shape (nel,).

        L (int): Harmonic band-limit.

        mm (int): Harmonic order :math:`m^\prime` at which to slice the matrix.

        polar (Tuple): Functions of :math:`\beta` and the meeting point from
            :func:`polar_terms`.

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
    unscale = xp.where(scale == 0, 1.0, xp.where(scale == -1, 1 / SCALE_FACTOR, 0.0))
    value = xp.where(active, sign * d_next * unscale, 0.0)

    # Rescale both iterates by an exact power of two when they leave [2^-400, 2^400]
    largest = xp.maximum(xp.abs(d_curr), xp.abs(d_next))
    up = largest > SCALE_LIMIT
    down = (largest < 1 / SCALE_LIMIT) & (largest > 0)
    factor = xp.where(up, 1 / SCALE_FACTOR, xp.where(down, SCALE_FACTOR, 1.0))
    d_prev = xp.where(active, d_curr * factor, d_prev)
    d_curr = xp.where(active, d_next * factor, d_curr)
    scale = xp.where(active, scale + up - down, scale)
    return value, (d_prev, d_curr, scale)


def polar_terms(beta, mm, xp):
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
