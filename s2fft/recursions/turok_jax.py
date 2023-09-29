import numpy as np
import jax.lax as lax
from jax import jit
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=(2, 3))
def compute_slice(beta: float, el: int, L: int, mm: int) -> jnp.ndarray:
    r"""Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok &
    Bucher recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`. The Turok & Bucher recursion is analytically correct
    from :math:`-\ell < m < \ell` however numerically it can become unstable for
    :math:`m > 0`. To avoid this we compute :math:`d_{m,
    m^{\prime}}^{\ell}(\beta)` for negative :math:`m` and then evaluate
    :math:`d_{m, -m^{\prime}}^{\ell}(\beta) = (-1)^{m-m^{\prime}} d_{-m,
    m^{\prime}}^{\ell}(\beta)` which we can again evaluate using a Turok &
    Bucher recursion.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 - m]`. This implementation has computational
    scaling :math:`\mathcal{O}(L)` and typically requires :math:`\sim 2L`
    operations.

    Args:
        beta (float): Polar angle in radians.

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension [2L-1].
    """
    dl = jnp.zeros(2 * L - 1, dtype=jnp.float64)
    dl = lax.cond(
        jnp.abs(beta) < 1e-10, lambda x: _north_pole(x, el, L, mm), lambda x: x, dl
    )
    dl = lax.cond(
        jnp.abs(beta - jnp.pi) < 1e-10,
        lambda x: _south_pole(x, el, L, mm),
        lambda x: x,
        dl,
    )
    dl = lax.cond(el == 0, lambda x: _el0(x, L), lambda x: x, dl)
    dl = lax.cond(
        jnp.any(dl),
        lambda x: x,
        lambda x: _compute_quarter_slice(x, beta, el, L, mm),
        dl,
    )

    return reindex(dl, el, L, mm)


@partial(jit, static_argnums=(3, 4))
def _compute_quarter_slice(
    dl: jnp.array, beta: float, el: int, L: int, mm: int
) -> jnp.ndarray:
    r"""Compute a single slice at :math:`m^{\prime}` of the Wigner-d matrix evaluated
    at :math:`\beta`.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1, 2L-1).

        beta (float): Polar angle in radians.

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1] populated only on the mm
            slice.
    """
    # These constants handle overflow by retrospectively renormalising
    big_const = 1e10
    bigi = 1.0 / big_const
    lbig = jnp.log(big_const)

    # Trigonometric constant adopted throughout
    c = jnp.cos(beta)
    s = jnp.sin(beta)
    t = jnp.tan(-beta / 2.0)
    lt = jnp.log(jnp.abs(t))
    c2 = jnp.cos(beta / 2.0)
    omc = 1.0 - c

    # Indexing boundaries
    half_slices = jnp.array([el + mm + 1, el - mm + 1], dtype=jnp.int64)
    lims = np.array([0, -1], dtype=np.int64)

    # Vectors with indexing -L < m < L adopted throughout

    # Populate vectors for first row
    def log_first_row_iteration(log_first_row_i_minus_1, i):
        ratio = (2 * el + 2 - i - 1) / i
        log_first_row_i = log_first_row_i_minus_1 + jnp.log(ratio) / 2 + lt
        return log_first_row_i, log_first_row_i

    log_first_row_0 = 2 * el * jnp.log(abs(c2))
    _, log_first_row_1_to_L_plus_abs_mm = lax.scan(
        log_first_row_iteration, log_first_row_0, np.arange(1, L + abs(mm))
    )
    log_first_row = jnp.concatenate(
        (
            jnp.atleast_1d(log_first_row_0),
            log_first_row_1_to_L_plus_abs_mm,
            jnp.zeros(L + 1 - abs(mm)),
        )
    )

    sign = (t / abs(t)) ** ((half_slices - 1) % 2)

    # Initialising coefficients cp(m)= cplus(l-m).
    m = jnp.arange(L)
    cpi = jnp.concatenate((2 / ((m + 1) * (2 * el - m)) ** 0.5, jnp.zeros(1)))
    cp2 = jnp.concatenate((jnp.zeros(1), cpi[1:] / cpi[:-1]))

    # Use Turok & Bucher recursion to evaluate a single half row
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    em = jnp.arange(1, L + 1)
    lrenorm = jnp.zeros(2, dtype=jnp.float64)

    # Static array of indices for first dimension of dl array
    indices = jnp.arange(2 * L - 1)

    for i in range(2):
        sgn = (-1) ** (i)

        # Initialise the vector
        dl = dl.at[lims[i]].set(1.0)
        lamb = ((el + 1) * omc - half_slices[i] + c) / s
        dl = dl.at[lims[i] + sgn].set(lamb * dl[lims[i]] * cpi[0])

        def renorm_iteration(m, dl_lrenorm):
            dl, lrenorm = dl_lrenorm
            lamb = ((el + 1) * omc - half_slices[i] + m * c) / s
            dl = dl.at[lims[i] + sgn * m].set(
                lamb * cpi[m - 1] * dl[lims[i] + sgn * (m - 1)]
                - cp2[m - 1] * dl[lims[i] + sgn * (m - 2)]
            )
            condition = dl[lims[i] + sgn * m] > big_const
            lrenorm = lax.cond(
                condition, lambda x: x.at[i].add(-lbig), lambda x: x, lrenorm
            )
            dl = lax.cond(
                condition,
                # multiply first m elements (if i == 0) or last m elements (if i == 1)
                # of dl array by bigi - use jnp.where rather than directly updating
                # array using 'in-place' update such as
                #     dl.at[lims[i]:lims[i] + sgn * (m + 1):sgn].multiply(bigi)
                # to avoid non-static array slice (due to m dependence) that will raise
                # an IndexError exception when used with lax.fori_loop
                lambda x: jnp.where((indices < (m + 1))[::sgn], bigi * x, x),
                lambda x: x,
                dl,
            )
            return dl, lrenorm

        dl, lrenorm = lax.fori_loop(2, L, renorm_iteration, (dl, lrenorm))

        # Apply renormalisation
        renorm = sign[i] * jnp.exp(log_first_row[half_slices[i] - 1] - lrenorm[i])

        if i == 0:
            dl = dl.at[: L - 1].multiply(renorm)

        if i == 1:
            dl = dl.at[-em].multiply((-1) ** ((mm - em + el + 1) % 2) * renorm)

    return jnp.nan_to_num(dl, neginf=0, posinf=0)


@partial(jit, static_argnums=(2, 3))
def _north_pole(dl, el, L, mm) -> jnp.ndarray:
    r"""Compute Wigner-d matrix for edge case where theta index is located on the
    north pole.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1).

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1].
    """
    dl = dl.at[:].set(0)
    if mm < 0:
        dl = dl.at[el + mm].set(1)
    else:
        dl = dl.at[2 * L - 2 - el + mm].set(1)
    return dl


@partial(jit, static_argnums=(2, 3))
def _south_pole(dl, el, L, mm) -> jnp.ndarray:
    r"""Compute Wigner-d matrix for edge case where theta index is located on the
    south pole.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1).

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1].
    """
    dl = dl.at[:].set(0)
    if mm > 0:
        dl = dl.at[el - mm].set((-1) ** (el + mm))
    else:
        dl = dl.at[2 * L - 2 - el - mm].set((-1) ** (el + mm))
    return dl


@partial(jit, static_argnums=(1))
def _el0(dl, L) -> jnp.ndarray:
    r"""Compute Wigner-d matrix for edge case where the harmonic degree is 0
    (monopole term).

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1).

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1].
    """
    dl = dl.at[:].set(0)
    dl = dl.at[-1].set(1)
    return dl


@partial(jit, static_argnums=(2, 3))
def reindex(dl, el, L, mm) -> jnp.ndarray:
    r"""Reorders indexing of Wigner-d matrix.

    Reindexes the Wigner-d matrix to centre m values around L-1.
    The original indexing is given by
    :math:`[-m \rightarrow -1, \dots, 0 \rightarrow m]` and the
    resulting indexing is given by
    :math:`[\dots, -m \rightarrow 0 \rightarrow m, \dots]`, where
    :math:`\dots` represents entries in which the values should be
    ignored. These extra entries are necessary to ensure :func:`~compute_slice`
    can operate with static length.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1).

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1].
    """
    dl = dl.at[: L - 1].set(jnp.roll(dl, L - el - 1)[: L - 1])
    dl = dl.at[L - 1 :].set(jnp.roll(dl, -(L - el - 1))[L - 1 :])

    m = jnp.arange(-L + 1, L + 1)
    dl = dl.at[L - 1 + m].multiply((-1) ** ((mm - m) % 2))

    return dl
