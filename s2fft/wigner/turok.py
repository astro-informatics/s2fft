import numpy as np


def compute_full(beta: float, L: int) -> np.ndarray:
    """Constructs the Wigner-d matrix via Turok recursion

    Args:

        beta (float): polar angle in radians
        L (int): Angular bandlimit

    Raises:

        ValueError: If polar angle is greater than pi

    Returns:

        Wigner-d matrix of dimension [2*L-1, 2*L-1]
    """
    if beta > np.pi:
        raise ValueError(f"Polar angle {beta} cannot be greater than pi")

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    el = L - 1
    return turok_fill(turok_quarter(dl, beta, el), el)


def turok_quarter(dl: np.ndarray, beta: float, l: int) -> np.ndarray:
    """Evaluates the left quarter triangle of the Wigner-d matrix via
        Turok recursion

    Args:
        dl (np.ndarray): Wigner-d matrix to populate
        beta (float): polar angle in radians
        l (int): Angular bandlimit - 1 (conventions)

    Returns:

        Wigner-d matrix of dimension [2*L-1, 2*L-1] with
        left quarter triangle populated.
    """
    # If beta < 0 dl = identity
    if np.abs(beta) < 0:
        return np.identity(2 * l + 1, dtype=np.float64)

    # Define constants adopted throughout
    lp1 = 1  # Offset for indexing (currently -L < m < L in 2D)

    # TODO: Can you remember the point of these big consts?
    big_const = 1.0
    big = big_const
    bigi = 1.0 / big_const
    lbig = np.log(big)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    si = 1.0 / s
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
    for m in range(1, l + 1):
        xm = l - m
        cpi[m - 1] = 2.0 / np.sqrt(l * (l + 1) - xm * (xm + 1))
        cp[m - 1] = 1.0 / cpi[m - 1]

    for m in range(2, l + 1):
        cp2[m - 1] = cpi[m - 1] * cp[m - 2]

    dl[1 - lp1, 1 - lp1] = 1.0
    dl[2 * l + 1 - lp1, 1 - lp1] = 1.0

    # Use Turok recursion to fill from diagonal to horizontal (lower left eight)
    for index in range(2, l + 2):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((l + 1) * omc - index + c) * si
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
                    for im in range(m + 1):
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


def turok_fill(dl: np.ndarray, l: int) -> np.ndarray:
    """Reflects Turok Wigner-d quarter plane to complete matrix

    Args:
        dl (np.ndarray): Wigner-d matrix to populate
        l (int): Angular bandlimit - 1 (conventions)

    Returns:

        Complete Wigner-d matrix of dimension [2*L-1, 2*L-1]
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
