import numpy as np
from jax import jit, lax
import jax.numpy as jnp
from functools import partial


def init(dl: np.ndarray, L: int, implementation: str = "vectorized") -> np.ndarray:
    r"""Initialise Wigner-d at argument :math:`\pi/2` for :math:`\ell=0` for
    Trapani & Navaza recursion (multiple implementations).

    Args:
        dl (np.ndarray): Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` allocated
            for all :math:`-L < m, m^\prime < L`, to be indexed by
            `dl[m + L - 1, m' + L - 1]`.

        L (int): Harmonic band-limit.

        implementation (str, optional): Implementation to adopt.  Supported
            implementations include {"loop", "vectorized", "jax"}.  Defaults to
            "vectorized".

    Returns:
        np.ndarray: Plane of Wigner-d initialised for :math:`\ell=0`,
    """

    if implementation.lower() == "loop":
        return init_nonjax(dl, L)

    elif implementation == "vectorized":
        return init_nonjax(dl, L)

    elif implementation == "jax":
        return init_jax(dl, L)

    else:
        raise ValueError(f"Implementation {implementation} not supported")


def init_nonjax(dl: np.ndarray, L: int) -> np.ndarray:
    r"""Initialise Wigner-d at argument :math:`\pi/2` for :math:`\ell=0` for
    Trapani & Navaza recursion (loop-based/vectorized implementation).

    See :func:`~init` for further details.

    Note:
        Loop-based/vectorized implementation.

    Args:
        dl (np.ndarray): Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` allocated
            for all :math:`-L < m, m^\prime < L`, to be indexed by
            `dl[m + L - 1, m' + L - 1]`.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Plane of Wigner-d initialised for :math:`\ell=0`,
    """

    el = 0
    dl[el + L - 1, el + L - 1] = 1.0

    return dl


@partial(jit, static_argnums=(1,))
def init_jax(dl: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""Initialise Wigner-d at argument :math:`\pi/2` for :math:`\ell=0` for
    Trapani & Navaza recursion (JAX implementation).

    See :func:`~init` for further details.

    Note:
        JAX implementation.

    Args:
        dl (jnp.ndarray): Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` allocated
            for all :math:`-L < m, m^\prime < L`, to be indexed by
            `dl[m + L - 1, m' + L - 1]`.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Plane of Wigner-d initialised for :math:`\ell=0`,
    """

    el = 0
    dl = dl.at[el + L - 1, el + L - 1].set(1.0)

    return dl


def compute_eighth(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for eighth of plane using
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
    r"""Compute Wigner-d at argument :math:`\pi/2` for quarter of plane using
    Trapani & Navaza recursion (vector implementation).

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. For :math:`\ell = 1` the recusion must already be
    initialised (see :func:`~init`).

    The Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` is indexed for
    :math:`-L < m, m^\prime < L` by `dl[m + L - 1, m' + L - 1]` but is only
    computed for the quarter of the plane
    :math:`0 <= m, m^\prime <= \ell`.

    Note:
        Vectorized implementation.

        For vectorised implementations it is better to compute the full quarter of the
        plane directly, rather than compute an eight and fill the quarter by symmetry.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with quarter of plane computed.
    """

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
def compute_quarter_jax(dl: jnp.ndarray, L: int, el: int) -> jnp.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for quarter of plane using
    Trapani & Navaza recursion (JAX implementation).

    The Wigner-d plane is computed by recursion over :math:`\ell` (`el`).
    Thus, for :math:`\ell > 0` the plane must be computed already for
    :math:`\ell - 1`. For :math:`\ell = 1` the recusion must already be
    initialised (see :func:`~init`).

    The Wigner-d plane :math:`d^\ell_{mm^\prime}(\pi/2)` is indexed for
    :math:`-L < m, m^\prime < L` by `dl[m + L - 1, m' + L - 1]` but is only
    computed for the quarter of the plane
    :math:`0 <= m, m^\prime <= \ell`.

    Note:
        JAX implementation.

        For vectorised implementations it is better to compute the full quarter of the
        plane directly, rather than compute an eight and fill the quarter by symmetry.

    Warning:
        Writes garbage outside of `m`,`mm` range for given `el`.

    Args:
        dl (jnp.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        jnp.ndarray: Plane of Wigner-d for `el`, with quarter of plane computed.
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

    def compute_dl_submatrix_slice(dl_slice_1_dl_slice_2, t1_fact_i_t2_fact_i):
        (
            t1_fact_i,
            t2_fact_i,
        ) = t1_fact_i_t2_fact_i
        dl_slice_1, dl_slice_2 = dl_slice_1_dl_slice_2
        t1 = 2 * mm / t1_fact_i * dl_slice_1
        t2 = t2_fact_i * dl_slice_2
        dl_slice_0 = t1 - t2
        return (dl_slice_0, dl_slice_1), dl_slice_0

    _, dl_submatrix = lax.scan(
        compute_dl_submatrix_slice,
        (dl[el - 1 + (L - 1), mm + (L - 1)], dl[el + (L - 1), mm + (L - 1)]),
        (t1_fact, t2_fact),
    )

    dl = dl.at[ms[:, None] + (L - 1), mm[None] + (L - 1)].set(dl_submatrix)

    return dl


def fill_eighth2quarter(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    r"""Fill in quarter of Wigner-d plane from eighth.

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
    r"""Fill in half of Wigner-d plane from quarter.

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
    r"""Fill in half of Wigner-d plane from quarter (vectorised implementation).

    See :func:`~fill_quarter2half` for further details.

    Note:
        Vectorized implementation.

    Args:
        dl (np.ndarray): Quarter of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with half of plane computed.
    """

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
    r"""Fill in half of Wigner-d plane from quarter (JAX implementation).

    See :func:`~fill_quarter2half` for further details.

    Note:
        JAX implementation.

    Args:
        dl (jnp.ndarray): Quarter of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        jnp.ndarray: Plane of Wigner-d for `el`, with half of plane computed.
    """

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
    r"""Fill in full Wigner-d plane from half.

    The Wigner-d plane passed as an argument should be computed for the half
    of the plane :math:`-\ell <= m <= \ell` and :math:`0 <= m^\prime <= \ell`.
    The returned plane is computed by symmetry for
    :math:`-\ell <= m, m^\prime <= \ell`.

    Note:
        Loop-based implementation.

    Args:
        dl (np.ndarray): Half of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

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
    r"""Fill in full Wigner-d plane from half (vectorized implementation).

    See :func:`~fill_half2full` for further details.

    Note:
        Vectorized implementation.

    Args:
        dl (np.ndarray): Half of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.
    """

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
    r"""Fill in full Wigner-d plane from half (JAX implementation).

    See :func:`~fill_half2full` for further details.

    Note:
        JAX implementation.

    Args:
        dl (jnp.ndarray): Half of Wigner-d plane for :math:`\ell` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        jnp.ndarray: Plane of Wigner-d for `el`, with full plane computed.
    """

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


def compute_full(
    dl: np.ndarray, L: int, el: int, implementation: str = "vectorized"
) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for full plane using
    Trapani & Navaza recursion (multiple implementations).

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

        implementation (str, optional): Implementation to adopt.  Supported
            implementations include {"loop", "vectorized", "jax"}.  Defaults to
            "vectorized".

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.

    """

    if implementation.lower() == "loop":
        return compute_full_loop(dl, L, el)

    elif implementation == "vectorized":
        return compute_full_vectorized(dl, L, el)

    elif implementation == "jax":
        return compute_full_jax(dl, L, el)

    else:
        raise ValueError(f"Implementation {implementation} not supported")


def compute_full_loop(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for full plane using
    Trapani & Navaza recursion (loop-based implementation).

    See :func:`~compute_full` for further details.

    Note:
        Loop-based implementation.

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.

    """

    _arg_checks(dl, L, el)

    dl = compute_eighth(dl, L, el)
    dl = fill_eighth2quarter(dl, L, el)
    dl = fill_quarter2half(dl, L, el)
    dl = fill_half2full(dl, L, el)

    return dl


def compute_quarter(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for quarter plane using
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
    Symmetry relations are then used to fill in the remainder of the quarter plane
    (see :func:`~fill_eighth2quarter`).

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with quarter plane computed.

    """

    _arg_checks(dl, L, el)

    dl = compute_eighth(dl, L, el)
    dl = fill_eighth2quarter(dl, L, el)

    return dl


def compute_full_vectorized(dl: np.ndarray, L: int, el: int) -> np.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for full plane using
    Trapani & Navaza recursion (vectorized implementation).

    See :func:`~compute_full` for further details.

    Note:
        Vectorized implementation.

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.

    """

    _arg_checks(dl, L, el)

    dl = compute_quarter_vectorized(dl, L, el)
    dl = fill_quarter2half_vectorized(dl, L, el)
    dl = fill_half2full_vectorized(dl, L, el)

    return dl


@partial(jit, static_argnums=(1,))
def compute_full_jax(dl: jnp.ndarray, L: int, el: int) -> jnp.ndarray:
    r"""Compute Wigner-d at argument :math:`\pi/2` for full plane using
    Trapani & Navaza recursion (JAX implementation).

    See :func:`~compute_full` for further details.

    Note:
        JAX implementation.

    Warning:
        This recursion may not be stable above :math:`\ell \gtrsim 1024`.

    Args:
        dl (np.ndarray): Wigner-d plane for :math:`\ell - 1` at :math:`\pi/2`.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.

    Returns:
        np.ndarray: Plane of Wigner-d for `el`, with full plane computed.

    """
    _arg_checks(dl, L, el)

    dl = compute_quarter_jax(dl, L, el)
    dl = fill_quarter2half_jax(dl, L, el)
    dl = fill_half2full_jax(dl, L, el)

    return dl


def _arg_checks(dl: np.ndarray, L: int, el: int):
    """Check arguments of Trapani functions.

    Args:
        dl (np.ndarray): Wigner-d plane of which to check shape.

        L (int): Harmonic band-limit.

        el (int): Spherical harmonic degree :math:`\ell`.
    """

    # assert 0 < el < L
    # assert dl.shape[0] == dl.shape[1] == 2 * L - 1
    # if L > 1024:
    #     logs.warning_log("Trapani recursion may not be stable for L > 1024")
