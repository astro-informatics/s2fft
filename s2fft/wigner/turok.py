from re import I
import numpy as np


def compute_full(beta: float, el: int, L: int) -> np.ndarray:
    """Constructs the complete Wigner-d matrix at polar angle
    :math:`\beta` using Turok recursion.

    The Wigner-d plane for a given :math:`\ell` (`el`) and :math:`\beta`
    is computed recursively over :math:`m, mm` labelled 'm' and 'mm'
    respectively.

    The Wigner-d plane :math:`d^\ell_{m, mm}(\beta)` is indexed for
    :math:`-L < m, mm < L` by `dl[m + L - 1, mm + L - 1]` but is only
    computed for the eighth of the plane
    :math:`mm <= m < \ell, 0 <= mm <= \ell`.
    Symmetry relations can be used to fill in the remainder of the plane if
    required (see :func:`~_fill`).

    Args:

        beta (float): Polar angle in radians.
        el (int): Harmonic degree of wigner-d matrix.
        L (int): Harmonic bandlimit of overall transform.

    Raises:

        ValueError: If el is greater than L.

    Returns:

        Wigner-d matrix of dimension [2*L-1, 2*L-1].
    """
    if el >= L:
        raise ValueError(
            f"Wigner-d bandlimit {el} cannot be equal to or greater than L={L}"
        )

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl[L - 1 - el : L + el, L - 1 - el : L + el] = _fill(turok_quarter(beta, el), el)
    return dl


def compute_slice(
    beta: float, el: int, L: int, mm: int, accelerate: bool = False
) -> np.ndarray:
    """Constructs a particular slice `mm` of the complete Wigner-d matrix
    at polar angle :math:`\beta` using Turok recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta`
    is computed recursively over :math:`m` labelled 'm' at a specific 'mm'.
    Depending on the value of :math:`mm`, one or more recursions over :math:`m`
    may be required to evaluate the Wigner-d slice.

    The Wigner-d slice :math:`d^\ell_{m, mm}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[m + L - 1]`. Symmetry relations are used to
    fill in the remainder of the plane if required (see :func:`~_fill`).
    If accelerate is True minimal symmetry reflections are computed to increase
    computational efficiency.

    Args:

        beta (float): Polar angle in radians.
        el (int): Harmonic degree of wigner-d matrix.
        L (int): Harmonic bandlimit of overall transform.
        mm (int): Harmonic degree at which to slice the matrix.
        accelerate (bool): Optimise indexing to minimise reflections.

    Raises:

        ValueError: If el is greater than L.
        ValueError: If el is less than mm.

    Returns:

        Wigner-d matrix mm slice of dimension [2*L-1].
    """
    if el < mm:
        raise ValueError(f"Wigner-D not valid for l={el} < mm={mm}.")

    if el >= L:
        raise ValueError(
            f"Wigner-d bandlimit {el} cannot be equal to or greater than L={L}"
        )

    dl = np.zeros(2 * L - 1, dtype=np.float64)
    dl[L - 1 - el : L + el] = turok_quarter_slice(beta, el, mm, accelerate)

    return dl


def turok_quarter_slice(
    beta: float, el: int, mm: int, accelerate: bool = False
) -> np.ndarray:
    """Evaluates the left quarter triangle of the Wigner-d matrix via
        Turok recursion at ONLY a specific spin index

    Args:
        beta (float): Polar angle in radians.
        l (int): Harmonic degree of Wigner-d matrix.
        mm (int): Harmonic degree at which to slice the matrix.
        accelerate (bool): Optimise indexing to minimise reflections.

    Returns:

        Wigner-d matrix slice of dimension [2*L-1] populated only on mm.
    """
    # Analytically evaluate singularities
    if np.isclose(beta, 0, atol=1e-8):
        dl = np.zeros(2 * el + 1, dtype=np.float64)
        dl[el - mm] = 1
        return dl

    if np.isclose(beta, np.pi, atol=1e-8):
        dl = np.zeros(2 * el + 1, dtype=np.float64)
        dl[el + mm] = (-1) ** (el + mm)
        return dl

    if el == 0:
        return 1

    # Define constants adopted throughout
    dl = np.zeros((2 * el + 1, 2 * el + 1), dtype=np.float64)
    lp1 = 1  # Offset for indexing (currently -l <= m <= l)

    # These constants handle overflow by retrospectively renormalising
    big_const = 1e10
    bigi = 1.0 / big_const
    lbig = np.log(big_const)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    t = np.tan(-beta / 2.0)
    c2 = np.cos(beta / 2.0)
    omc = 1.0 - c

    # Vectors with indexing -L < m < L adopted throughout
    lrenorm = np.zeros(2 * el + 1, dtype=np.float64)
    cpi = np.zeros(2 * el + 1, dtype=np.float64)
    cp2 = np.zeros(2 * el + 1, dtype=np.float64)
    log_first_row = np.zeros(2 * el + 1, dtype=np.float64)
    sign = np.zeros(2 * el + 1, dtype=np.float64)

    # Populate vectors for first row
    log_first_row[0] = 2.0 * el * np.log(np.abs(c2))
    sign[0] = 1.0

    for i in range(2, 2 * el + 2):
        m = el + 1 - i
        ratio = np.sqrt((m + el + 1) / (el - m))
        log_first_row[i - 1] = log_first_row[i - 2] + np.log(ratio) + np.log(np.abs(t))
        sign[i - 1] = sign[i - 2] * t / np.abs(t)

    # Initialising coefficients cp(m)= cplus(l-m).
    for m in range(1, el + 2):
        xm = el - m
        cpi[m - 1] = 2.0 / np.sqrt(el * (el + 1) - xm * (xm + 1))

    for m in range(2, el + 2):
        cp2[m - 1] = cpi[m - 1] / cpi[m - 2]

    dl[1 - lp1, 1 - lp1] = 1.0
    dl[2 * el + 1 - lp1, 1 - lp1] = 1.0

    # Use Turok recursion to fill from diagonal to horizontal (lower left eight)
    index = el - mm + lp1
    m_cap = el - np.abs(mm) + lp1

    for i in range(el - np.abs(mm) + lp1, el + 2):
        dl[i - lp1, 1 - lp1] = 1.0
        lamb = ((el + 1) * omc - i + c) / s
        dl[i - lp1, 2 - lp1] = lamb * dl[i - lp1, 1 - lp1] * cpi[0]

        if i > 2:
            for m in range(2, i):
                lamb = ((el + 1) * omc - i + m * c) / s
                dl[i - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[i - lp1, m - lp1]
                    - cp2[m - 1] * dl[i - lp1, m - 1 - lp1]
                )

                if dl[i - lp1, m + 1 - lp1] > big_const:
                    lrenorm[i - 1] = lrenorm[i - 1] - lbig
                    for im in range(1, m + 2):
                        dl[i - lp1, im - lp1] = dl[i - lp1, im - lp1] * bigi

    for i in range(el + 2, el + np.abs(mm) + 2):
        dl[i - lp1, 1 - lp1] = 1.0
        lamb = ((el + 1) * omc - i + c) / s
        dl[i - lp1, 2 - lp1] = lamb * dl[i - lp1, 1 - lp1] * cpi[0]

        if i < 2 * el:
            for m in range(2, 2 * el - i + 2):
                lamb = ((el + 1) * omc - i + m * c) / s
                dl[i - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[i - lp1, m - lp1]
                    - cp2[m - 1] * dl[i - lp1, m - 1 - lp1]
                )

                if dl[i - lp1, m + 1 - lp1] > big_const:
                    lrenorm[i - 1] = lrenorm[i - 1] - lbig
                    for im in range(1, m + 2):
                        dl[i - lp1, im - lp1] = dl[i - lp1, im - lp1] * bigi

    # Apply renormalisation
    for i in range(el - np.abs(mm) + lp1, el + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    for i in range(el + 2, el + np.abs(mm) + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, 2 * el + 2 - i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    if accelerate == True:
        # Reflect across diagonal
        if mm >= 0:
            for i in range(index, index + 1):
                sgn = -1
                for j in range(i + 1, el + 2):
                    dl[i - lp1, j - lp1] = dl[j - lp1, i - lp1] * sgn
                    sgn = sgn * (-1)

        # Reflect across anti-diagonal
        if mm < 0:
            for i in range(el + mm + 1, el + mm + 2):
                for j in range(el + 1, 2 * el + 1 - i + 1):
                    dl[2 * el + 2 - i - lp1, 2 * el + 2 - j - lp1] = dl[
                        j - lp1, i - lp1
                    ]

        # Conjugate reflect across m=0
        for i in range(m_cap):
            dl[index - lp1, 2 * el - i] = (-1) ** (float(mm + i + 1)) * dl[el + mm, i]

        if np.abs(mm) > 0:
            if mm >= 0:
                step = 1
                for i in range(m_cap, el):
                    dl[index - lp1, 2 * el - i] = (-1) ** (mm + i + 1) * dl[
                        el + mm - step, m_cap - 1
                    ]
                    step += 1
            else:
                step = 2 * np.abs(mm) - 1
                for i in range(m_cap, el):
                    dl[index - lp1, 2 * el - i] = dl[el - mm - step, m_cap - 1]
                    step -= 1

        # Finally invert the appropriate elements
        if mm > 0:
            dl[index - lp1, el + 1 :] *= (-1) ** (el + 1)
        elif mm < 0:
            dl[index - lp1, el + np.abs(mm) :] *= (-1) ** (el + 1)
        else:
            dl[index - lp1, el:] *= (-1) ** (el + 1)

    else:
        dl = _fill(dl, el)

    return dl[index - lp1]


def turok_quarter(beta: float, l: int) -> np.ndarray:
    """Evaluates the left quarter triangle of the Wigner-d matrix via
        Turok recursion

    Args:
        beta (float): Polar angle in radians.
        l (int): Harmonic degree of Wigner-d matrix.

    Returns:

        Wigner-d matrix of dimension [2*L-1, 2*L-1] with
        left quarter triangle populated.
    """
    # Analytically evaluate singularities
    if np.isclose(beta, 0, atol=1e-8):
        return np.identity(2 * l + 1, dtype=np.float64)

    if np.isclose(beta, np.pi, atol=1e-8):
        dl = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.float64)
        for m in range(-l, l + 1):
            dl[l - m, l + m] = (-1) ** (l + m)
        return dl

    if l == 0:
        return 1

    # Define constants adopted throughout
    dl = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.float64)
    lp1 = 1  # Offset for indexing (currently -L < m < L in 2D)

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
    lrenorm = np.zeros(2 * l + 1, dtype=np.float64)
    cp = np.zeros(2 * l + 1, dtype=np.float64)
    cpi = np.zeros(2 * l + 1, dtype=np.float64)
    cp2 = np.zeros(2 * l + 1, dtype=np.float64)
    log_first_row = np.zeros(2 * l + 1, dtype=np.float64)
    sign = np.zeros(2 * l + 1, dtype=np.float64)

    # Populate vectors for first row
    log_first_row[0] = 2.0 * l * np.log(np.abs(c2))
    sign[0] = 1.0

    for i in range(2, 2 * l + 2):
        m = l + 1 - i
        ratio = np.sqrt((m + l + 1) / (l - m))
        log_first_row[i - 1] = log_first_row[i - 2] + np.log(ratio) + np.log(np.abs(t))
        sign[i - 1] = sign[i - 2] * t / np.abs(t)

    # Initialising coefficients cp(m)= cplus(l-m).
    for m in range(1, l + 2):
        xm = l - m
        cpi[m - 1] = 2.0 / np.sqrt(l * (l + 1) - xm * (xm + 1))
        cp[m - 1] = 1.0 / cpi[m - 1]

    for m in range(2, l + 2):
        cp2[m - 1] = cpi[m - 1] * cp[m - 2]

    dl[1 - lp1, 1 - lp1] = 1.0
    dl[2 * l + 1 - lp1, 1 - lp1] = 1.0

    # Use Turok recursion to fill from diagonal to horizontal (lower left eight)
    for index in range(2, l + 2):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((l + 1) * omc - index + c) / s
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index > 2:
            for m in range(2, index):
                lamb = ((l + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )

                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(1, m + 2):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    # Use Turok recursion to fill horizontal to anti-diagonal (upper left eight)
    for index in range(l + 2, 2 * l + 1):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((l + 1) * omc - index + c) / s
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index < 2 * l:
            for m in range(2, 2 * l - index + 2):
                lamb = ((l + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )
                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(1, m + 2):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    # Apply renormalisation
    for i in range(1, l + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    for i in range(l + 2, 2 * l + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, 2 * l + 2 - i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    return dl


def _fill(dl: np.ndarray, l: int) -> np.ndarray:
    """Reflects Turok Wigner-d quarter plane to complete matrix

    Args:
        dl (np.ndarray): Wigner-d matrix to populate by symmetry.
        l (int): Harmonic degree of Wigner-d matrix.

    Returns:

        Complete Wigner-d matrix of dimension [2*L-1, 2*L-1].
    """
    lp1 = 1  # Offset for indexing (currently -L < m < L in 2D)

    # Reflect across anti-diagonal
    for i in range(1, l + 1):
        for j in range(l + 1, 2 * l + 1 - i + 1):
            dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1] = dl[j - lp1, i - lp1]

    # Reflect across diagonal
    for i in range(1, l + 2):
        sgn = -1
        for j in range(i + 1, l + 2):
            dl[i - lp1, j - lp1] = dl[j - lp1, i - lp1] * sgn
            sgn = sgn * (-1)

    # Fill right matrix
    for i in range(l + 2, 2 * l + 2):
        sgn = (-1) ** (i + 1)
        for j in range(1, 2 * l + 2 - i + 1):
            dl[j - lp1, i - lp1] = dl[i - lp1, j - lp1] * sgn
            sgn = sgn * (-1)

        for j in range(i, 2 * l + 2):
            dl[j - lp1, i - lp1] = dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1]

    for i in range(l + 2, 2 * l + 2):
        for j in range(2 * l + 3 - i, i - 1 + 1):
            dl[j - lp1, i - lp1] = dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1]

    return dl
