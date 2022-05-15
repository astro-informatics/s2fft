import numpy as np
from jax import jit
import jax.numpy as jnp
from functools import partial
import logs


def init(dl: np.ndarray, L: int) -> np.ndarray:
    """Initialise Wigner-d at argument :math:`\pi/2` for :math:`\ell=0` for
    Trapani & Navaza recursion.
    """

    el = 0
    dl[el + L - 1, el + L - 1] = 1.0

    return dl


@partial(jit, static_argnums=(1,))
def init_jax(dl: jnp.ndarray, L: int) -> jnp.ndarray:
    """TODO"""

    el = 0
    dl = dl.at[el + L - 1, el + L - 1].set(1.0)

    return dl


def compute_eighth(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """Compute Wigner-d at argument :math:`\pi/2` for eighth of plane using
    Trapani & Navaza recursion.

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. For :math:`\ell = 1` the recusion must already be
    initialised (see :func:`~init`).

    The Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` is indexed for
    :math:`-L < m, m^\prime < L` by `dl[m + L - 1, m' + L - 1]` but is only
    computed for the eighth of the plane
    :math:`m^\prime <= m < \ell` and :math:`0 <= m^\prime <= \ell`.
    Symmetry relations can be used to fill in the remainder of the plane if
    required (see :func:`~fill_eighth2quarter`, :func:`~fill_quarter2half`,
    :func:`~fill_half2full`).

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Note:
        Loop-based implementation.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with eighth of plane computed.

    """

    _arg_checks(dl, L, el)

    dmm = np.zeros(L)

    # Equation (9) of T&N (2006).
    dmm[0] = -np.sqrt((2 * el - 1) / (2 * el)) * dl[el - 1 + (L - 1), 0 + (L - 1)]

    # Equation (10) of T&N (2006).
    for mm in range(1, el + 1):  # 1:el
        dmm[mm] = (
            np.sqrt(el / 2 * (2 * el - 1) / (el + mm) / (el + mm - 1))
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
            / np.sqrt((el - m) * (el + m + 1))
            * dl[m + 1 + (L - 1), mm + (L - 1)]
        )

        # Remaining m cases.
        for m in range(el - 2, mm - 1, -1):  # el-2:-1:mm
            t1 = (
                2
                * mm
                / np.sqrt((el - m) * (el + m + 1))
                * dl[m + 1 + (L - 1), mm + (L - 1)]
            )
            t2 = (
                np.sqrt((el - m - 1) * (el + m + 2) / (el - m) / (el + m + 1))
                * dl[m + 2 + (L - 1), mm + (L - 1)]
            )
            dl[m + (L - 1), mm + (L - 1)] = t1 - t2

    return dl


def compute_quarter_vectorized(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    dmm = np.zeros(L)

    # Equation (9) of T&N (2006).
    dmm[0] = -np.sqrt((2 * el - 1) / (2 * el)) * dl[el - 1 + (L - 1), 0 + (L - 1)]

    # Equation (10) of T&N (2006).
    mm = np.arange(1, el + 1)
    dmm[mm] = (
        np.sqrt(el / 2 * (2 * el - 1) / (el + mm) / (el + mm - 1))
        * dl[el - 1 + (L - 1), mm - 1 + (L - 1)]
    )

    # Initialise dl for next el.
    mm = np.arange(el + 1)
    dl[el + (L - 1), mm + (L - 1)] = dmm[mm]

    # Equation (11) of T&N (2006).
    # m = el-1 case (t2 = 0).
    m = el - 1
    dl[m + (L - 1), mm + (L - 1)] = (
        2 * mm / np.sqrt((el - m) * (el + m + 1)) * dl[m + 1 + (L - 1), mm + (L - 1)]
    )

    # Equation (11) of T&N (2006).
    # Remaining m cases.
    ms = np.arange(el - 2, -1, -1)
    t1_fact = np.sqrt((el - ms) * (el + ms + 1))
    t2_fact = np.sqrt((el - ms - 1) * (el + ms + 2) / (el - ms) / (el + ms + 1))
    for i, m in enumerate(ms):  # compute quarter plane since vectorizes
        t1 = 2 * mm / t1_fact[i] * dl[m + 1 + (L - 1), mm + (L - 1)]
        t2 = t2_fact[i] * dl[m + 2 + (L - 1), mm + (L - 1)]
        dl[m + (L - 1), mm + (L - 1)] = t1 - t2

    return dl


@partial(jit, static_argnums=(1,))
def compute_quarter_jax(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """TODO

    writes garbage outside of m,mm range for given el

    """

    _arg_checks(dl, L, el)

    dmm = jnp.zeros(L)

    # Equation (9) of T&N (2006).
    dmm = dmm.at[0].set(
        -jnp.sqrt((2 * el - 1) / (2 * el)) * dl[el - 1 + (L - 1), 0 + (L - 1)]
    )

    # Equation (10) of T&N (2006).
    mm = jnp.arange(1, L)
    dmm = dmm.at[mm].set(
        jnp.sqrt(el / 2 * (2 * el - 1) / (el + mm) / (el + mm - 1))
        * dl[el - 1 + (L - 1), mm - 1 + (L - 1)]
    )

    # Initialise dl for next el.
    mm = jnp.arange(L)
    dl = dl.at[el + (L - 1), mm + (L - 1)].set(dmm[mm])

    # Equation (11) of T&N (2006).
    # m = el-1 case (t2 = 0).
    m = el - 1
    dl = dl.at[m + (L - 1), mm + (L - 1)].set(
        2 * mm / jnp.sqrt((el - m) * (el + m + 1)) * dl[m + 1 + (L - 1), mm + (L - 1)]
    )

    # Equation (11) of T&N (2006).
    # Remaining m cases.
    # ms = jnp.arange(el - 2, -1, -1)
    ms = jnp.arange((L - 1) - 2, -1, -1) - (L - 1) + el
    ms_clip = jnp.where(ms < 0, 0, ms)
    t1_fact = jnp.sqrt((el - ms_clip) * (el + ms_clip + 1))
    t2_fact = jnp.sqrt(
        (el - ms_clip - 1) * (el + ms_clip + 2) / (el - ms_clip) / (el + ms_clip + 1)
    )
    for i, m in enumerate(ms):  # compute quarter plane since vectorizes
        t1 = 2 * mm / t1_fact[i] * dl[m + 1 + (L - 1), mm + (L - 1)]
        t2 = t2_fact[i] * dl[m + 2 + (L - 1), mm + (L - 1)]
        dl = dl.at[m + (L - 1), mm + (L - 1)].set(t1 - t2)

    return dl


def fill_eighth2quarter(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """Fill in quarter of Wigner-d plane from eighth.

    The Wigner-d plane passed as an argument should be computed for the eighth
    of the plane  :math:`m^\prime <= m < \ell` and :math:`0 <= m^\prime <= \ell`.
    The returned plane is computed by symmetry for
    :math:`0 <= m, m^\prime <= \ell`.

    Args:
        dl (np.ndarray): Eighth of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with quarter of plane computed.
    """

    _arg_checks(dl, L, el)

    # Diagonal symmetry to fill in quarter.
    for m in range(el + 1):  # 0:el
        for mm in range(m + 1, el + 1):  # m+1:el
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (m + mm) * dl[
                mm + (L - 1), m + (L - 1)
            ]

    return dl


def fill_quarter2half(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """Fill in half of Wigner-d plane from quarter.

    The Wigner-d plane passed as an argument should be computed for the quarter
    of the plane :math:`0 <= m, m^\prime <= \ell`.  The
    returned plane is computed by symmetry for
    :math:`-\ell <= m <= \ell` and :math:`0 <= m^\prime <= \ell`.

    Args:

        dl (np.ndarray): Quarter of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with half of plane computed.
    """

    _arg_checks(dl, L, el)

    # Symmetry in m to fill in half.
    for mm in range(0, el + 1):  # 0:el
        for m in range(-el, 0):  # -el:-1
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (el + mm) * dl[
                -m + (L - 1), mm + (L - 1)
            ]

    return dl


def fill_quarter2half_vectorized(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    # Symmetry in m to fill in half.
    mm = np.arange(0, el + 1)  # 0:el
    m = np.arange(-el, 0)  # -el:-1
    m_grid, mm_grid = np.meshgrid(m, mm)

    dl[m_grid + (L - 1), mm_grid + (L - 1)] = (-1) ** (el + mm_grid) * dl[
        -m_grid + (L - 1), mm_grid + (L - 1)
    ]

    return dl


@partial(jit, static_argnums=(1,))
def fill_quarter2half_jax(dl: jnp.ndarray, L: int, el: int) -> jnp.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    # Symmetry in m to fill in half.
    # mm = jnp.arange(0, el + 1)  # 0:el
    # m = jnp.arange(-el, 0)  # -el:-1
    mm = jnp.arange(0, L)  # 0:el
    m = jnp.arange(-(L - 1), 0)  # -el:-1
    m_grid, mm_grid = jnp.meshgrid(m, mm)

    dl = dl.at[m_grid + (L - 1), mm_grid + (L - 1)].set(
        (-1) ** (el + mm_grid) * dl[-m_grid + (L - 1), mm_grid + (L - 1)]
    )

    return dl


def fill_half2full(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """Fill in full Wigner-d plane from half.

    The Wigner-d plane passed as an argument should be computed for the half
    of the plane :math:`-\ell <= m <= \ell` and :math:`0 <= m^\prime <= \ell`.
    The returned plane is computed by symmetry for
    :math:`-\ell <= m, m^\prime <= \ell`.

    Args:
        dl (np.ndarray): Quarter of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.
    """

    _arg_checks(dl, L, el)

    # Symmetry in mm to fill in remaining plane.
    for mm in range(-el, 0):  # -el:-1
        for m in range(-el, el + 1):  # -el:el
            dl[m + (L - 1), mm + (L - 1)] = (-1) ** (el + abs(m)) * dl[
                m + (L - 1), -mm + (L - 1)
            ]

    return dl


def fill_half2full_vectorized(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    # Symmetry in mm to fill in remaining plane.
    mm = np.arange(-el, 0)
    m = np.arange(-el, el + 1)
    m_grid, mm_grid = np.meshgrid(m, mm)

    dl[m_grid + (L - 1), mm_grid + (L - 1)] = (-1) ** (el + abs(m_grid)) * dl[
        m_grid + (L - 1), -mm_grid + (L - 1)
    ]

    return dl


@partial(jit, static_argnums=(1,))
def fill_half2full_jax(dl: jnp.ndarray, L: int, el: int) -> jnp.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    # Symmetry in mm to fill in remaining plane.
    # mm = jnp.arange(-el, 0)
    # m = jnp.arange(-el, el + 1)
    mm = jnp.arange(-(L - 1), 0)
    m = jnp.arange(-(L - 1), L)
    m_grid, mm_grid = jnp.meshgrid(m, mm)

    dl = dl.at[m_grid + (L - 1), mm_grid + (L - 1)].set(
        (-1) ** (el + abs(m_grid)) * dl[m_grid + (L - 1), -mm_grid + (L - 1)]
    )

    return dl


def compute_full(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """Compute Wigner-d at argument :math:`\pi/2` for full plane using
    Trapani & Navaza recursion.

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. For :math:`\ell = 1` the recusion must already be
    initialised (see :func:`~init`).

    The Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` (`el`) is indexed for
    :math:`-L < m, m^\prime < L` by `dl[m + L - 1, m' + L - 1]`. The plane is
    computed directly for the eighth of the plane
    :math:`m^\prime <= m < \ell` and :math:`0 <= m^\prime <= \ell`
    (see :func:`~compute_eighth`).
    Symmetry relations are then used to fill in the remainder of the plane
    (see :func:`~fill_eighth2quarter`,
    :func:`~fill_quarter2half`, :func:`~fill_half2full`).

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with eighth of plane computed.

    """

    _arg_checks(dl, L, el)

    dl = compute_eighth(dl, L, el)
    dl = fill_eighth2quarter(dl, L, el)
    dl = fill_quarter2half(dl, L, el)
    dl = fill_half2full(dl, L, el)

    return dl


def compute_quarter(dl: np.ndarray, L: int, el: int) -> np.ndarray:

    _arg_checks(dl, L, el)

    dl = compute_eighth(dl, L, el)
    dl = fill_eighth2quarter(dl, L, el)

    return dl


def compute_full_vectorized(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    dl = compute_quarter_vectorized(dl, L, el)
    dl = fill_quarter2half_vectorized(dl, L, el)
    dl = fill_half2full_vectorized(dl, L, el)

    return dl


@partial(jit, static_argnums=(1,))
def compute_full_jax(dl: jnp.ndarray, L: int, el: int) -> jnp.ndarray:
    """TODO"""

    _arg_checks(dl, L, el)

    dl = compute_quarter_jax(dl, L, el)
    dl = fill_quarter2half_jax(dl, L, el)
    dl = fill_half2full_jax(dl, L, el)

    return dl


def _arg_checks(dl: np.ndarray, L: int, el: int):
    """Check arguments of Trapani functions.

    Args:

        dl: Wigner-d plane to check shape of.

        L: Harmonic band-limit.

        el: Spherical harmonic degree :math:`\ell`.
    """

    # assert 0 < el < L
    # assert dl.shape[0] == dl.shape[1] == 2 * L - 1
    # if L > 1024:
    #     logs.warning_log("Trapani recursion may not be stable for L > 1024")
