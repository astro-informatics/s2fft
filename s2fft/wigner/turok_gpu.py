import numpy as np
import jax.lax as lax
from jax import jit
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=(1, 2, 3))
def compute_slice(beta: float, el: int, L: int, mm: int) -> jnp.ndarray:
    r"""Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta`
    is computed recursively over :math:`m` labelled 'm' at a specific :math:`m^{\prime}`.
    The Turok & Bucher recursion is analytically correct from :math:`-\ell < m < \ell`
    however numerically it can become unstable for :math:`m > 0`. To avoid this we
    compute :math:`d_{m, m^{\prime}}^{\ell}(\beta)` for negative :math:`m` and then evaluate
    :math:`d_{m, -m^{\prime}}^{\ell}(\beta) = (-1)^{m-m^{\prime}} d_{-m, m^{\prime}}^{\ell}(\beta)`
    which we can again evaluate using a Turok & Bucher recursion.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 - m]`. This implementation has computational
    scaling :math:`\mathcal{O}(L)` and typically requires :math:`\sim 2L` operations.

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

    return _reindex(dl, el, L)


@partial(jit, static_argnums=(3, 4))
def _compute_quarter_slice(
    dl: jnp.array, beta: float, el: int, L: int, mm: int
) -> jnp.ndarray:
    r"""Compute a single slice at :math:`m^{\prime}` of the Wigner-d matrix evaluated
    at :math:`\beta`.

    Args:
        dl (np.ndarray): Wigner-d matrix to populate (shape: 2L-1, 2L-1).

        beta (float): Polar angle in radians.

        l (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix slice of dimension [2L-1] populated only on the mm slice.
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
    half_slices = jnp.zeros(2, dtype=jnp.int16)
    half_slices = half_slices.at[0].set(el + mm + 1)
    half_slices = half_slices.at[1].set(el - mm + 1)

    lims = jnp.zeros(2, dtype=jnp.int16)
    lims = lims.at[0].set(0)
    lims = lims.at[1].set(-1)

    # Vectors with indexing -L < m < L adopted throughout
    lrenorm = jnp.zeros(2, dtype=jnp.float64)
    sign = jnp.zeros(2, dtype=jnp.float64)
    cpi = jnp.zeros(L + 1, dtype=jnp.float64)
    cp2 = jnp.zeros(L + 1, dtype=jnp.float64)
    log_first_row = jnp.zeros(2 * L + 1, dtype=jnp.float64)

    # Populate vectors for first row
    log_first_row = log_first_row.at[0].set(2.0 * el * jnp.log(jnp.abs(c2)))

    for i in range(2, L + 1 + np.abs(mm)):
        ratio = (2 * el + 2 - i) / (i - 1)
        log_first_row = log_first_row.at[i - 1].set(
            log_first_row[i - 2] + jnp.log(ratio) / 2 + lt
        )

    sign = sign.at[0].set((t / jnp.abs(t)) ** ((half_slices[0] - 1) % 2))
    sign = sign.at[1].set((t / jnp.abs(t)) ** ((half_slices[1] - 1) % 2))

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi = cpi.at[0].set(2.0 / jnp.sqrt(2 * el))
    for m in range(2, L + 1):
        cpi = cpi.at[m - 1].set(2.0 / jnp.sqrt(m * (2 * el + 1 - m)))
        cp2 = cp2.at[m - 1].set(cpi[m - 1] / cpi[m - 2])

    # Use Turok & Bucher recursion to evaluate a single half row
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    em = jnp.arange(1, L + 1)

    for i in range(2):
        sgn = (-1) ** (i)

        # Initialise the vector
        dl = dl.at[lims[i]].set(1.0)
        lamb = ((el + 1) * omc - half_slices[i] + c) / s
        dl = dl.at[lims[i] + sgn].set(lamb * dl[lims[i]] * cpi[0])

        for m in range(2, L):
            lamb = ((el + 1) * omc - half_slices[i] + m * c) / s
            dl = dl.at[lims[i] + sgn * m].set(
                lamb * cpi[m - 1] * dl[lims[i] + sgn * (m - 1)]
                - cp2[m - 1] * dl[lims[i] + sgn * (m - 2)]
            )
            lrenorm = lax.cond(
                dl[lims[i] + sgn * m] > big_const,
                lambda x: _increment_normalisation(x, i, lbig),
                lambda x: x,
                lrenorm,
            )
            dl = lax.cond(
                dl[lims[i] + sgn * m] > big_const,
                lambda x: _renormalise(x, lims, sgn, i, m, bigi),
                lambda x: x,
                dl,
            )

        # Apply renormalisation
        renorm = sign[i] * jnp.exp(log_first_row[half_slices[i] - 1] - lrenorm[i])

        if i == 0:
            dl = dl.at[: L - 1].multiply(renorm)

        if i == 1:
            dl = dl.at[-em].multiply((-1) ** ((mm - em + el + 1) % 2) * renorm)

    return jnp.nan_to_num(dl, neginf=0, posinf=0)


@partial(jit, static_argnums=(4))
def _renormalise(dl, lims, sgn, i, m, bigi) -> jnp.ndarray:
    for im in range(m + 1):
        dl = dl.at[lims[i] + sgn * im].multiply(bigi)
    return dl


@partial(jit)
def _increment_normalisation(lrenorm, i, lbig) -> jnp.ndarray:
    lrenorm = lrenorm.at[i].set(lrenorm[i] - lbig)
    return lrenorm


@partial(jit, static_argnums=(2, 3))
def _north_pole(dl, el, L, mm) -> jnp.ndarray:
    dl = dl.at[:].set(0)
    if mm < 0:
        dl = dl.at[el + mm].set(1)
    else:
        dl = dl.at[2 * L - 2 - el + mm].set(1)
    return dl


@partial(jit, static_argnums=(2, 3))
def _south_pole(dl, el, L, mm) -> jnp.ndarray:
    dl = dl.at[:].set(0)
    if mm > 0:
        dl = dl.at[el - mm].set((-1) ** (el + mm))
    else:
        dl = dl.at[2 * L - 2 - el - mm].set((-1) ** (el + mm))
    return dl


@partial(jit, static_argnums=(1))
def _el0(dl, L) -> jnp.ndarray:
    dl = dl.at[:].set(0)
    dl = dl.at[-1].set(1)
    return dl


@partial(jit, static_argnums=(2))
def _reindex(dl, el, L) -> jnp.ndarray:
    dl = dl.at[: L - 1].set(jnp.roll(dl, L - el - 1)[: L - 1])
    return dl.at[L - 1 :].set(jnp.roll(dl, -(L - el - 1))[L - 1 :])
