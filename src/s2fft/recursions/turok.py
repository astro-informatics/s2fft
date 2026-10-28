import math
from warnings import warn

import numpy as np

from s2fft.recursions.price_mcewen import (
    _SCALE_BITS,
    _SCALE_FACTOR,
    _SCALE_LIMIT,
    _start_values_scalar,
)


def compute_full(beta: float, el: int, L: int) -> np.ndarray:
    r"""
    Compute the complete Wigner-d matrix at polar angle :math:`\beta` using
    Turok & Bucher recursion.

    The Wigner-d plane for a given :math:`\ell` (`el`) and :math:`\beta`
    is computed recursively over :math:`m, m^{\prime}`.

    The Wigner-d plane :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m, m^{\prime} < L` by dl[L - 1 + :math:`m`, L - 1 + :math:`m^{\prime}`]
    but is only computed for the eighth of the plane
    :math:`m^{\prime} <= m < \ell, 0 <= m^{\prime} <= \ell`.
    Symmetry relations are used to fill in the remainder of the plane (see
    :func:`~fill`).

    Args:
        beta (float): Polar angle in radians.

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

    Raises:
        ValueError: If el is greater than L.

    Returns:
        np.ndarray: Wigner-d matrix of dimension [2L-1, 2L-1].

    """
    if el >= L:
        raise ValueError(
            f"Wigner-d bandlimit {el} cannot be equal to or greater than L={L}"
        )
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = compute_quarter(dl, beta, el, L)
    return fill(dl, el, L)


def compute_slice(
    beta: float, el: int, L: int, mm: int, positive_m_only: bool = False
) -> np.ndarray:
    r"""
    Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher
    recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta` is computed
    recursively over :math:`m` labelled 'm' at a specific :math:`m^{\prime}`, by two
    half recursions running inwards from the edges :math:`m = \mp\ell`. Each half
    recursion is only stable while the slice grows in its direction, so the halves meet
    at :math:`m^\star = \mathrm{round}(m^\prime\cos\beta)`, the centre of the
    oscillatory region of the slice, rather than at :math:`m = 0`; near the poles the
    slice peaks at :math:`m = \pm m^\prime` and decays away from it. The starting
    values are evaluated directly and the iterates rescaled by exact powers of two, as in
    :func:`~s2fft.recursions.price_mcewen.compute_all_slices`.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for :math:`-L <
    m < L` by `dl[L - 1 + m]`. This implementation has computational scaling
    :math:`\mathcal{O}(L)` and typically requires :math:`\sim 2L` operations.

    Args:
        beta (float): Polar angle in radians.

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

        positive_m_only (bool, optional): Compute Wigner-d matrix for slice at m greater
            than zero only.  Defaults to False.

        Whether to exploit conjugate symmetry. By construction
            this only leads to significant improvement for mm = 0. Defaults to False.

    Raises:
        ValueError: If el is greater than L.

        ValueError: If el is less than mm.

        Warning: If positive_m_only is true but mm not 0.

    Returns:
        np.ndarray: Wigner-d matrix mm slice of dimension [2L-1].

    """
    if el < mm:
        raise ValueError(f"Wigner-D not valid for l={el} < mm={mm}.")

    if el >= L:
        raise ValueError(
            f"Wigner-d bandlimit {el} cannot be equal to or greater than L={L}"
        )

    if positive_m_only and mm != 0:
        positive_m_only = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform.",
            stacklevel=2,
        )

    dl = np.zeros(2 * L - 1, dtype=np.float64)
    return compute_quarter_slice(dl, beta, el, L, mm, positive_m_only)


def compute_quarter_slice(
    dl: np.ndarray,
    beta: float,
    el: int,
    L: int,
    mm: int,
    positive_m_only: bool = False,
) -> np.ndarray:
    r"""
    Compute a single slice at :math:`m^{\prime}` of the Wigner-d matrix evaluated
    at :math:`\beta`.

    Args:
        dl (np.ndarray): Wigner-d matrix slice to populate (shape: 2L-1).

        beta (float): Polar angle in radians.

        l (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

        positive_m_only (bool, optional): Compute Wigner-d matrix for slice at m greater
            than zero only.  Defaults to False.

    Returns:
        np.ndarray: Wigner-d matrix slice of dimension [2L-1] populated only on the mm slice.

    """
    # Analytically evaluate singularities
    if np.isclose(beta, 0, atol=1e-8):
        dl[L - 1 + mm] = 1
        return dl

    if np.isclose(beta, np.pi, atol=1e-8):
        dl[L - 1 - mm] = (-1) ** (el + mm)
        return dl

    if el == 0:
        dl[L - 1] = 1
        return dl

    # Starting values at the edges m = -el and m = el, as mantissas and exponents
    starts = _start_values_scalar(beta, el, mm)

    # Trigonometric constant adopted throughout
    c = math.cos(beta)
    s = math.sin(beta)
    # m_c cos(beta) + mm_i through half angles, as in price_mcewen._recursion_step
    hemisphere = 1.0 if c >= 0 else -1.0
    half_angle = 2 * (math.sin(beta / 2) if c >= 0 else math.cos(beta / 2)) ** 2

    # The two half recursions meet at the centre of the oscillatory region of the slice
    m_star = int(np.rint(mm * c))

    for i in range(2):
        if not (positive_m_only and i == 0):
            mm_i = mm if i == 0 else -mm
            n_rows = m_star + el if i == 0 else el - m_star + 1
            # Iterates within [2^-400, 2^400] with an integer scale index k, the value
            # being d 2^(800 k), as in price_mcewen._recursion_step
            mantissa, exponent = starts[i]
            scale = round(exponent / _SCALE_BITS)
            d_prev, d_curr = 0.0, 0.0
            cpi_prev = 0.0
            for q in range(n_rows):
                if q == 0:
                    d_next = math.ldexp(mantissa, exponent - _SCALE_BITS * scale)
                else:
                    m_c = el + 1 - q
                    lamb = (m_c + hemisphere * mm_i) - m_c * half_angle
                    lamb = -hemisphere * lamb / s
                    cpi = 2.0 / math.sqrt(q * (2 * el + 1 - q))
                    cp2 = cpi / cpi_prev if q >= 2 else 0.0
                    cpi_prev = cpi
                    d_next = cpi * lamb * d_curr - cp2 * d_prev

                m = -el + q if i == 0 else el - q
                sign = 1 - 2 * (((m if i == 0 else mm) + el) % 2)
                if scale == 0:
                    dl[L - 1 + m] = sign * d_next
                elif scale == -1:
                    dl[L - 1 + m] = sign * d_next / _SCALE_FACTOR

                # Rescale the iterates by an exact power of two when out of range
                d_prev, d_curr = d_curr, d_next
                largest = max(abs(d_prev), abs(d_curr))
                if largest > _SCALE_LIMIT:
                    d_prev, d_curr = d_prev / _SCALE_FACTOR, d_curr / _SCALE_FACTOR
                    scale += 1
                elif 0 < largest < 1 / _SCALE_LIMIT:
                    d_prev, d_curr = d_prev * _SCALE_FACTOR, d_curr * _SCALE_FACTOR
                    scale -= 1

    return dl


def compute_quarter(dl: np.ndarray, beta: float, ell: int, L: int) -> np.ndarray:  # noqa: C901
    """
    Compute the left quarter triangle of the Wigner-d matrix via Turok & Bucher
    recursion.

    Args:
        dl (np.ndarray): Wigner-d matrix slice to populate (shape: 2L-1, 2L-1).

        beta (float): Polar angle in radians.

        ell (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Wigner-d matrix of dimension [2L-1, 2L-1] with left quarter
        triangle populated.

    """
    # Analytically evaluate singularities
    if np.isclose(beta, 0, atol=1e-8):
        dl[L - 1 - ell : L + ell, L - 1 - ell : L + ell] = np.identity(
            2 * ell + 1, dtype=np.float64
        )
        return dl

    if np.isclose(beta, np.pi, atol=1e-8):
        for m in range(-ell, ell + 1):
            dl[L - 1 - m, L - 1 + m] = (-1) ** (ell + m)
        return dl

    if ell == 0:
        dl[L - 1, L - 1] = 1
        return dl

    # Define constants adopted throughout
    lp1 = 1 - (L - 1 - ell)  # Offset for indexing (currently -L < m < L in 2D)

    # These constants handle overflow by retrospectively renormalising
    big_const = 1e10
    big = big_const
    bigi = 1.0 / big_const
    lbig = np.log(big)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    t = np.tan(-beta / 2.0)
    c2 = np.cos(beta / 2.0)
    omc = 1.0 - c

    # Vectors with indexing -L < m < L adopted throughout
    lrenorm = np.zeros(2 * ell + 1, dtype=np.float64)
    cp = np.zeros(2 * ell + 1, dtype=np.float64)
    cpi = np.zeros(2 * ell + 1, dtype=np.float64)
    cp2 = np.zeros(2 * ell + 1, dtype=np.float64)
    log_first_row = np.zeros(2 * ell + 1, dtype=np.float64)
    sign = np.zeros(2 * ell + 1, dtype=np.float64)

    # Populate vectors for first row
    log_first_row[0] = 2.0 * ell * np.log(np.abs(c2))
    sign[0] = 1.0

    for i in range(2, 2 * ell + 2):
        m = ell + 1 - i
        ratio = np.sqrt((m + ell + 1) / (ell - m))
        log_first_row[i - 1] = log_first_row[i - 2] + np.log(ratio) + np.log(np.abs(t))
        sign[i - 1] = sign[i - 2] * t / np.abs(t)

    # Initialising coefficients cp(m)= cplus(l-m).
    for m in range(1, ell + 2):
        xm = ell - m
        cpi[m - 1] = 2.0 / np.sqrt(ell * (ell + 1) - xm * (xm + 1))
        cp[m - 1] = 1.0 / cpi[m - 1]

    for m in range(2, ell + 2):
        cp2[m - 1] = cpi[m - 1] * cp[m - 2]

    dl[1 - lp1, 1 - lp1] = 1.0
    dl[2 * ell + 1 - lp1, 1 - lp1] = 1.0

    # Use Turok & Bucher recursion to fill from diagonal to horizontal (lower left eight)
    for index in range(2, ell + 2):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((ell + 1) * omc - index + c) / s
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index > 2:
            for m in range(2, index):
                lamb = ((ell + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )

                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(1, m + 2):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    # Use Turok & Bucher recursion to fill horizontal to anti-diagonal (upper left eight)
    for index in range(ell + 2, 2 * ell + 1):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((ell + 1) * omc - index + c) / s
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index < 2 * ell:
            for m in range(2, 2 * ell - index + 2):
                lamb = ((ell + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )
                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(1, m + 2):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    # Apply renormalisation
    for i in range(1, ell + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    for i in range(ell + 2, 2 * ell + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, 2 * ell + 2 - i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    return dl


def fill(dl: np.ndarray, ell: int, L: int) -> np.ndarray:
    """
    Reflects Wigner-d quarter plane to complete full matrix by using symmetry
    properties of the Wigner-d matrices.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate by symmetry.

        ell (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: A complete Wigner-d matrix of dimension [2L-1, 2L-1].

    """
    lp1 = 1 - (L - 1 - ell)  # Offset for indexing (currently -L < m < L in 2D)

    # Reflect across anti-diagonal
    for i in range(1, ell + 1):
        for j in range(ell + 1, 2 * ell + 1 - i + 1):
            dl[2 * ell + 2 - i - lp1, 2 * ell + 2 - j - lp1] = dl[j - lp1, i - lp1]

    # Reflect across diagonal
    for i in range(1, ell + 2):
        sgn = -1
        for j in range(i + 1, ell + 2):
            dl[i - lp1, j - lp1] = dl[j - lp1, i - lp1] * sgn
            sgn = sgn * (-1)

    # Fill right matrix
    for i in range(ell + 2, 2 * ell + 2):
        sgn = (-1) ** (i + 1)

        for j in range(1, 2 * ell + 2 - i + 1):
            dl[j - lp1, i - lp1] = dl[i - lp1, j - lp1] * sgn
            sgn = sgn * (-1)

        for j in range(i, 2 * ell + 2):
            dl[j - lp1, i - lp1] = dl[2 * ell + 2 - i - lp1, 2 * ell + 2 - j - lp1]

    for i in range(ell + 2, 2 * ell + 2):
        for j in range(2 * ell + 3 - i, i - 1 + 1):
            dl[j - lp1, i - lp1] = dl[2 * ell + 2 - i - lp1, 2 * ell + 2 - j - lp1]

    return dl
