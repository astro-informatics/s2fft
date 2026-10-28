r"""
Gauss-Legendre nodes, as polar angles, and weights.

The nodes :math:`\theta_k` are the zeros of :math:`P_L(\cos\theta)`. Taking them as
:math:`\arccos` of the roots :math:`x_k` loses accuracy near the poles: there a rounding of
:math:`x_k` near :math:`\pm 1` moves :math:`\theta_k` by about :math:`u/\sin\theta_k`.
Here the nodes are found by Newton's method in :math:`\theta` instead, with :math:`P_L`
evaluated through the differences :math:`D_\ell = P_\ell - P_{\ell-1}` in
:math:`s = \sin^2(\theta/2)`, so that no rounded :math:`x = 1 - 2s` is formed. A last
Newton step and the weights :math:`2/(dP_L/d\theta)^2` at the final nodes are evaluated in
double-double arithmetic and rounded once to float64. In every case tested, the angles and
the weights are then correctly rounded.
"""

import operator
from fractions import Fraction
from functools import lru_cache
from math import factorial

import numpy as np

# Double-double arithmetic (Dekker 1971): a value is the unevaluated sum hi + lo.
_SPLIT = 134217729.0  # 2**27 + 1


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def _two_prod(a, b):
    p = a * b
    a_hi = _SPLIT * a
    a_hi = a_hi - (a_hi - a)
    b_hi = _SPLIT * b
    b_hi = b_hi - (b_hi - b)
    a_lo, b_lo = a - a_hi, b - b_hi
    return p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def _dd_add(x, y):
    s, e = _two_sum(x[0], y[0])
    return _fast_two_sum(s, e + (x[1] + y[1]))


def _dd_mul(x, y):
    p, e = _two_prod(x[0], y[0])
    return _fast_two_sum(p, e + (x[0] * y[1] + x[1] * y[0]))


def _dd_scale(x, c):
    p, e = _two_prod(x[0], c)
    return _fast_two_sum(p, e + x[1] * c)


def _dd_div(x, c):
    q = x[0] / c
    p, e = _two_prod(q, c)
    return _fast_two_sum(q, ((x[0] - p) - e + x[1]) / c)


def _dd_div_dd(x, y):
    q = x[0] / y[0]
    r = _dd_add(x, _dd_scale(y, -q))
    return _fast_two_sum(q, r[0] / y[0])


def _dd_const(value: Fraction) -> tuple[float, float]:
    hi = float(value)
    return hi, float(value - Fraction(hi))


_PI = _dd_const(Fraction("3.14159265358979323846264338327950288419716939937510582097"))
# Taylor coefficients (-1)^n / (2n + 1)! of sin t / t, enough for |t| <= pi/4
_SIN_COEFFS = [_dd_const(Fraction((-1) ** n, factorial(2 * n + 1))) for n in range(18)]


def _dd_sin(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sin(t) in double-double, for float64 t with |t| <= pi/4."""
    t2 = _two_prod(t, t)
    acc = (np.full_like(t, _SIN_COEFFS[-1][0]), np.full_like(t, _SIN_COEFFS[-1][1]))
    for c_hi, c_lo in reversed(_SIN_COEFFS[:-1]):
        acc = _dd_add(_dd_mul(acc, t2), (np.full_like(t, c_hi), np.full_like(t, c_lo)))
    return _dd_scale(acc, t)


def _legendre_by_differences(s: np.ndarray, L: int):
    r"""
    :math:`P_L(1 - 2s)`, :math:`D_L = P_L - P_{L-1}` and :math:`dP_L/d\theta`, in float64.

    With :math:`x = 1 - 2s`, the three-term recurrence becomes
    :math:`(\ell + 1) D_{\ell+1} = \ell D_\ell - 2s(2\ell + 1) P_\ell`, in which :math:`s`
    enters only multiplicatively, so its relative accuracy is kept near the poles.
    """
    p, d = 1 - 2 * s, -2 * s
    for ell in range(1, L):
        d = (ell * d - 2 * s * (2 * ell + 1) * p) / (ell + 1)
        p = p + d
    # dP_L/dtheta = -sin(theta) P_L'(x), with x P_L - P_{L-1} = D_L - 2s P_L
    dpdt = L * (d - 2 * s * p) / (2 * np.sqrt(s * (1 - s)))
    return p, d, dpdt


def _legendre_by_differences_dd(s, L: int):
    """:math:`P_L` and :math:`D_L` as in :func:`_legendre_by_differences`, in double-double."""
    two_s = (2 * s[0], 2 * s[1])
    neg_two_s = (-two_s[0], -two_s[1])
    p = _dd_add((np.ones_like(s[0]), np.zeros_like(s[0])), neg_two_s)
    d = neg_two_s
    for ell in range(1, L):
        b = _dd_scale(_dd_mul(two_s, p), float(2 * ell + 1))
        d = _dd_div(_dd_add(_dd_scale(d, float(ell)), (-b[0], -b[1])), float(ell + 1))
        p = _dd_add(p, d)
    return p, d


def _weights_dd(s, p, d, delta, L: int) -> np.ndarray:
    r"""
    The weights :math:`2/(dP_L/d\theta)^2` at :math:`\theta + \delta`, rounded once to
    float64, from :math:`s`, :math:`P_L` and :math:`D_L` at :math:`\theta` in double-double.

    At a zero of :math:`P_L`, Legendre's equation gives
    :math:`d^2P_L/d\theta^2 = -\cot\theta \, dP_L/d\theta`, and differentiating it once more
    gives the third derivative, so the small step :math:`\delta` to the zero multiplies the
    weight by :math:`1 + 2\delta\cot\theta + (\cot^2\theta - 1 - L(L+1))\delta^2`, to second
    order.
    """
    one = (np.ones_like(s[0]), np.zeros_like(s[0]))
    # sin(theta) dP_L/dtheta = L (D_L - 2s P_L), and 2 sin^2(theta) = 8s (1 - s)
    sin_dpdt = _dd_scale(_dd_add(d, _dd_scale(_dd_mul(s, p), -2.0)), float(L))
    two_sin2 = _dd_scale(_dd_mul(s, _dd_add(one, (-s[0], -s[1]))), 8.0)
    w = _dd_div_dd(two_sin2, _dd_mul(sin_dpdt, sin_dpdt))
    cot = (1 - 2 * s[0]) / (2 * np.sqrt(s[0] * (1 - s[0])))
    scale = 2 * delta * cot + (cot * cot - 1 - L * (L + 1)) * delta * delta
    w = _dd_add(w, (scale * w[0], np.zeros_like(w[0])))
    return w[0] + w[1]


@lru_cache(maxsize=16)
def _nodes_and_weights(L: int) -> tuple[np.ndarray, np.ndarray]:
    if L < 1:
        raise ValueError(f"Gauss-Legendre quadrature needs at least one node, not {L}.")
    m = L // 2  # Nodes in the open northern hemisphere
    k = np.arange(1, m + 1)
    theta = np.pi * (4 * k - 1) / (4 * L + 2)
    for _ in range(50):
        p, _, dpdt = _legendre_by_differences(np.sin(theta / 2) ** 2, L)
        step = p / dpdt
        theta = theta - step
        # Stop within a few ulps: rounding sets a floor there, and the double-double
        # step below converges quadratically from it.
        if np.all(np.abs(step) <= 64 * np.spacing(theta)):
            break

    # A last Newton step with P_L in double-double: theta + delta is the root to
    # double-double accuracy, and two_sum rounds it to the nearest float64 once.
    sin_half = _dd_sin(theta / 2)
    s = _dd_mul(sin_half, sin_half)
    p, d = _legendre_by_differences_dd(s, L)
    dpdt = L * (d[0] - 2 * s[0] * p[0]) / (2 * np.sqrt(s[0] * (1 - s[0])))
    delta = -(p[0] / dpdt)
    north = _two_sum(theta, delta)
    south = _dd_add(
        (np.full_like(theta, _PI[0]), np.full_like(theta, _PI[1])),
        (-north[0], -north[1]),
    )
    weights = _weights_dd(s, p, d, delta, L)

    if L % 2:
        # The middle node is pi/2, where s = 1/2 exactly
        s = (np.array([0.5]), np.array([0.0]))
        p, d = _legendre_by_differences_dd(s, L)
        mid_theta, mid_weight = np.array([_PI[0] / 2]), _weights_dd(s, p, d, 0.0, L)
    else:
        mid_theta, mid_weight = np.array([]), np.array([])

    thetas = np.concatenate([north[0], mid_theta, (south[0] + south[1])[::-1]])
    weights = np.concatenate([weights, mid_weight, weights[::-1]])
    thetas.setflags(write=False)
    weights.setflags(write=False)
    return thetas, weights


def nodes_and_weights(L: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Gauss-Legendre nodes as polar angles, in ascending order, and their weights.

    Args:
        L (int): Number of nodes, the harmonic band-limit for GL sampling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The angles :math:`\theta_k`, the zeros of
        :math:`P_L(\cos\theta)`, and the weights :math:`w_k` of the quadrature on
        :math:`x = \cos\theta \in [-1, 1]`, each rounded once from double-double.

    """
    thetas, weights = _nodes_and_weights(operator.index(L))
    return thetas.copy(), weights.copy()
