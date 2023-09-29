import numpy as np
from warnings import warn


def compute_full(beta: float, el: int, L: int) -> np.ndarray:
    r"""Compute the complete Wigner-d matrix at polar angle :math:`\beta` using
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
    r"""Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher
    recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta` is computed
    recursively over :math:`m` labelled 'm' at a specific :math:`m^{\prime}`. The Turok
    & Bucher recursion is analytically correct from :math:`-\ell < m < \ell` however
    numerically it can become unstable for :math:`m > 0`. To avoid this we compute
    :math:`d_{m, m^{\prime}}^{\ell}(\beta)` for negative :math:`m` and then evaluate
    :math:`d_{m, -m^{\prime}}^{\ell}(\beta) = (-1)^{m-m^{\prime}} d_{-m,
    m^{\prime}}^{\ell}(\beta)` which we can again evaluate using a Turok & Bucher
    recursion.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for :math:`-L <
    m < L` by `dl[L - 1 - m]`. This implementation has computational scaling
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
            + "Defering to complex transform."
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
    r"""Compute a single slice at :math:`m^{\prime}` of the Wigner-d matrix evaluated
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

    # These constants handle overflow by retrospectively renormalising
    big_const = 1e10
    bigi = 1.0 / big_const
    lbig = np.log(big_const)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    t = np.tan(-beta / 2.0)
    lt = np.log(np.abs(t))
    c2 = np.cos(beta / 2.0)
    omc = 1.0 - c

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]
    lims = [L - 1 - el, L - 1 + el]

    # Vectors with indexing -L < m < L adopted throughout
    lrenorm = np.zeros(2, dtype=np.float64)
    sign = np.zeros(2, dtype=np.float64)
    cpi = np.zeros(el + 1, dtype=np.float64)
    cp2 = np.zeros(el + 1, dtype=np.float64)
    log_first_row = np.zeros(2 * el + 1, dtype=np.float64)

    # Populate vectors for first row
    log_first_row[0] = 2.0 * el * np.log(np.abs(c2))

    for i in range(2, np.max(half_slices) + 1):
        ratio = (2 * el + 2 - i) / (i - 1)
        log_first_row[i - 1] = log_first_row[i - 2] + np.log(ratio) / 2 + lt

    for i, slice in enumerate(half_slices):
        sign[i] = (t / np.abs(t)) ** ((slice - 1) % 2)

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi[0] = 2.0 / np.sqrt(2 * el)
    for m in range(2, el + 1):
        cpi[m - 1] = 2.0 / np.sqrt(m * (2 * el + 1 - m))
        cp2[m - 1] = cpi[m - 1] / cpi[m - 2]

    # Use Turok & Bucher recursion to evaluate a single half row
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    for i, slice in enumerate(half_slices):
        if not (positive_m_only and i == 0):
            sgn = (-1) ** (i)

            # Initialise the vector
            dl[lims[i]] = 1.0
            lamb = ((el + 1) * omc - slice + c) / s
            dl[lims[i] + sgn * 1] = lamb * dl[lims[i]] * cpi[0]

            for m in range(2, el + 1):
                lamb = ((el + 1) * omc - slice + m * c) / s
                dl[lims[i] + sgn * m] = (
                    lamb * cpi[m - 1] * dl[lims[i] + sgn * (m - 1)]
                    - cp2[m - 1] * dl[lims[i] + sgn * (m - 2)]
                )
                if dl[lims[i] + sgn * m] > big_const:
                    lrenorm[i] = lrenorm[i] - lbig
                    for im in range(m + 1):
                        dl[lims[i] + sgn * im] = dl[lims[i] + sgn * im] * bigi

            # Apply renormalisation
            renorm = sign[i] * np.exp(log_first_row[slice - 1] - lrenorm[i])

            if i == 0:
                for m in range(el):
                    dl[lims[i] + sgn * m] = dl[lims[i] + sgn * m] * renorm

            if i == 1:
                for m in range(el + 1):
                    dl[lims[i] + sgn * m] = (
                        (-1) ** ((mm - m + el) % 2) * dl[lims[i] + sgn * m] * renorm
                    )

    s_ind = 0 if positive_m_only else -el
    for m in range(s_ind, el + 1):
        dl[m + L - 1] *= (-1) ** (abs(mm - m))

    return dl


def compute_quarter(dl: np.ndarray, beta: float, l: int, L: int) -> np.ndarray:
    """Compute the left quarter triangle of the Wigner-d matrix via Turok & Bucher
    recursion.

    Args:
        dl (np.ndarray): Wigner-d matrix slice to populate (shape: 2L-1, 2L-1).

        beta (float): Polar angle in radians.

        l (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Wigner-d matrix of dimension [2L-1, 2L-1] with left quarter
        triangle populated.
    """
    # Analytically evaluate singularities
    if np.isclose(beta, 0, atol=1e-8):
        dl[L - 1 - l : L + l, L - 1 - l : L + l] = np.identity(
            2 * l + 1, dtype=np.float64
        )
        return dl

    if np.isclose(beta, np.pi, atol=1e-8):
        for m in range(-l, l + 1):
            dl[L - 1 - m, L - 1 + m] = (-1) ** (l + m)
        return dl

    if l == 0:
        dl[L - 1, L - 1] = 1
        return dl

    # Define constants adopted throughout
    lp1 = 1 - (L - 1 - l)  # Offset for indexing (currently -L < m < L in 2D)

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

    # Use Turok & Bucher recursion to fill from diagonal to horizontal (lower left eight)
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

    # Use Turok & Bucher recursion to fill horizontal to anti-diagonal (upper left eight)
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


def fill(dl: np.ndarray, l: int, L: int) -> np.ndarray:
    """Reflects Wigner-d quarter plane to complete full matrix by using symmetry
    properties of the Wigner-d matrices.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate by symmetry.

        l (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: A complete Wigner-d matrix of dimension [2L-1, 2L-1].
    """
    lp1 = 1 - (L - 1 - l)  # Offset for indexing (currently -L < m < L in 2D)

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
