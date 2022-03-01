import numpy as np

# @deprecated
def trapani_eighth_offset(dl, L: int, el: int):
    f"""Compute dl(pi/2) for eighth plane using Trapani recursion.

    Say some more stuff here.

    Args:

        L (int):

    Returns:

        ():

    """

    if el == 0:

        dl[el + L - 1, el + L - 1] = 1.0

    else:

        dmm = np.zeros(L)

        # Equation (9) of T&N (2006).
        dmm[0] = -np.sqrt((2 * el - 1) / (2 * el)) * dl[el - 1 + (L - 1), 0 + (L - 1)]

        # Equation (10) of T&N (2006).
        for mm in range(1, el + 1):  # 1:el
            dmm[mm] = (
                np.sqrt(el)
                / np.sqrt(2)
                * np.sqrt(2 * el - 1)
                / np.sqrt(el + mm)
                / np.sqrt(el + mm - 1)
                * dl[el - 1 + (L - 1), mm - 1 + (L - 1)]
            )

        # Initialise dl for next el.
        for mm in range(el + 1):  # 0:el
            dl[el + (L - 1), mm + (L - 1)] = dmm[mm]

        # Equation (11) of T&N (2006).
        for mm in range(el + 1):  # 0:el

            # m = el-1 case (t2 = 0).
            m = el - 1
            dl[m + (L - 1), mm + (L - 1)] = (
                2
                * mm
                / np.sqrt(el - m)
                / np.sqrt(el + m + 1)
                * dl[m + 1 + (L - 1), mm + (L - 1)]
            )

            # Remaining m cases.
            for m in range(el - 2, mm - 1, -1):  # el-2:-1:mm
                t1 = (
                    2
                    * mm
                    / np.sqrt(el - m)
                    / np.sqrt(el + m + 1)
                    * dl[m + 1 + (L - 1), mm + (L - 1)]
                )
                t2 = (
                    np.sqrt(el - m - 1)
                    * np.sqrt(el + m + 2)
                    / np.sqrt(el - m)
                    / np.sqrt(el + m + 1)
                    * dl[m + 2 + (L - 1), mm + (L - 1)]
                )
                dl[m + (L - 1), mm + (L - 1)] = t1 - t2

    return dl


def trapani_fill_offset_eighth2quarter(dl, L, el):

    # Diagonal symmetry to fill in quarter.
    for m in range(el + 1):  # 0:el
        for mm in range(m + 1, el + 1):  # m+1:el
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (m + mm) * dl[
                mm + (L - 1), m + (L - 1)
            ]

    return dl


def trapani_fill_offset_quarter2half(dl, L, el):

    # Symmetry in m to fill in half.
    for mm in range(0, el + 1):  # 0:el
        for m in range(-el, 0):  # -el:-1
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (el + mm) * dl[
                -m + (L - 1), mm + (L - 1)
            ]

    return dl


def trapani_fill_offset_half2full(dl, L, el):

    # Symmetry in mm to fill in remaining plane.
    for mm in range(-el, 0):  # -el:-1
        for m in range(-el, el + 1):  # -el:el
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (el + abs(m)) * dl[
                m + (L - 1), -mm + (L - 1)
            ]

    return dl


def trapani_full(dl, L: int, el: int):

    dl = trapani_eighth_offset(dl, L, el)
    dl = trapani_fill_offset_eighth2quarter(dl, L, el)
    dl = trapani_fill_offset_quarter2half(dl, L, el)
    dl = trapani_fill_offset_half2full(dl, L, el)

    return dl
