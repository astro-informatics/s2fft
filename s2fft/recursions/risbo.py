import numpy as np


def compute_full(dl: np.ndarray, beta: float, L: int, el: int) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\beta` for full plane using
    Risbo recursion.

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. At present, for :math:`\ell = 0` the recusion is initialised.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\beta`.

        beta (float): Argument :math:`\beta` at which to compute Wigner-d plane.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el` and `beta`, with full plane computed.
    """

    _arg_checks(dl, beta, L, el)

    if el == 0:
        el = 0
        dl[el + L - 1, el + L - 1] = 1.0

    elif el == 1:
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        coshb = np.cos(beta / 2.0)
        sinhb = np.sin(beta / 2.0)
        sqrt2 = np.sqrt(2.0)

        dl[-1 + L - 1, -1 + L - 1] = coshb**2
        dl[-1 + L - 1, 0 + L - 1] = sinb / sqrt2
        dl[-1 + L - 1, 1 + L - 1] = sinhb**2

        dl[0 + L - 1, -1 + L - 1] = -sinb / sqrt2
        dl[0 + L - 1, 0 + L - 1] = cosb
        dl[0 + L - 1, 1 + L - 1] = sinb / sqrt2

        dl[1 + L - 1, -1 + L - 1] = sinhb**2
        dl[1 + L - 1, 0 + L - 1] = -sinb / sqrt2
        dl[1 + L - 1, 1 + L - 1] = coshb**2

    else:
        coshb = -np.cos(beta / 2.0)
        sinhb = np.sin(beta / 2.0)

        # Initialise the plane of the dl-matrix to 0.0 for the recursion
        # from l - 1 to l - 1/2.
        dd = np.zeros((2 * el + 2, 2 * el + 2))
        j = 2 * el - 1
        rj = float(j)  # TODO: is this necessary?
        for k in range(0, j):
            sqrt_jmk = np.sqrt(j - k)
            sqrt_kp1 = np.sqrt(k + 1)

            for i in range(0, j):
                sqrt_jmi = np.sqrt(j - i)
                sqrt_ip1 = np.sqrt(i + 1)

                dlj = dl[k - (el - 1) + L - 1, i - (el - 1) + L - 1] / j

                dd[i, k] += sqrt_jmi * sqrt_jmk * dlj * coshb
                dd[i + 1, k] -= sqrt_ip1 * sqrt_jmk * dlj * sinhb
                dd[i, k + 1] += sqrt_jmi * sqrt_kp1 * dlj * sinhb
                dd[i + 1, k + 1] += sqrt_ip1 * sqrt_kp1 * dlj * coshb

        # Having constructed the d^(l+1/2) matrix in dd, do the second
        # half-step recursion from dd to dl. Start by initilalising
        # the plane of the dl-matrix to 0.0.
        dl[-el + L - 1 : el + 1 + L - 1, -el + L - 1 : el + 1 + L - 1] = 0.0
        j = 2 * el
        rj = float(j)  # TODO: is this necessary?
        for k in range(0, j):
            sqrt_jmk = np.sqrt(j - k)
            sqrt_kp1 = np.sqrt(k + 1)

            for i in range(0, j):
                sqrt_jmi = np.sqrt(j - i)
                sqrt_ip1 = np.sqrt(i + 1)

                ddj = dd[i, k] / j

                dl[k - el + L - 1, i - el + L - 1] += sqrt_jmi * sqrt_jmk * ddj * coshb
                dl[k - el + L - 1, i + 1 - el + L - 1] -= (
                    sqrt_ip1 * sqrt_jmk * ddj * sinhb
                )
                dl[k + 1 - el + L - 1, i - el + L - 1] += (
                    sqrt_jmi * sqrt_kp1 * ddj * sinhb
                )
                dl[k + 1 - el + L - 1, i + 1 - el + L - 1] += (
                    sqrt_ip1 * sqrt_kp1 * ddj * coshb
                )

    return dl


def _arg_checks(dl: np.ndarray, beta: float, L: int, el: int):
    """Check arguments of Risbo functions.

    Args:
        dl (np.ndarray): Wigner-d plane of which to check shape.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.
    """

    assert 0 <= el < L  # Should be < not <= once have init routine.
    assert dl.shape[0] == dl.shape[1] == 2 * L - 1
    assert 0 <= beta <= np.pi
