import numpy as np
import jax.lax as lax 
from jax import jit
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=(1, 2, 3))
def compute_slice(beta: float, el: int, L: int, mm: int
) -> jnp.ndarray:
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

    Raises:
        ValueError: If el is greater than L.

        ValueError: If el is less than mm.

        ValueError: If dl dimension is not 1.

        ValueError: If dl shape is incorrect.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension [2L-1].
    """
    if el < mm:
        raise ValueError(f"Wigner-D not valid for l={el} < mm={mm}.")

    if el >= L:
        raise ValueError(
            f"Wigner-d bandlimit {el} cannot be equal to or greater than L={L}"
        )

    dl = jnp.zeros(2 * L - 1, dtype=jnp.float64)
    
    dl = lax.cond(jnp.abs(beta) < 1e-8, lambda x: north_pole(x, L, mm), lambda x: x, dl)
    dl = lax.cond(jnp.abs(beta - jnp.pi) < 1e-8, lambda x: south_pole(x, L, el, mm), lambda x: x, dl)
    dl = lax.cond(el == 0, lambda x: el0(x, L), lambda x: x, dl)
    dl = lax.cond(jnp.any(dl), lambda x: x, lambda x: compute_quarter_slice(x, beta, el, L, mm), dl)

    return dl


@partial(jit, static_argnums=(2, 3, 4))
def compute_quarter_slice(dl: jnp.array, beta: float, el: int, L: int, mm: int
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
    lims = lims.at[0].set(L - 1 - el)
    lims = lims.at[1].set(L - 1 + el)

    # Vectors with indexing -L < m < L adopted throughout
    lrenorm = jnp.zeros(2, dtype=jnp.float64)
    sign = jnp.zeros(2, dtype=jnp.float64)
    cpi = jnp.zeros(el + 1, dtype=jnp.float64)
    cp2 = jnp.zeros(el + 1, dtype=jnp.float64)
    log_first_row = jnp.zeros(2 * L + 1, dtype=jnp.float64)

    # Populate vectors for first row
    log_first_row = log_first_row.at[0].set(2.0 * el * jnp.log(jnp.abs(c2)))

    for i in range(2, el + np.abs(mm) + 2):
        ratio = (2 * el + 2 - i) / (i - 1)
        log_first_row = log_first_row.at[i - 1].set(
            log_first_row[i - 2] + jnp.log(ratio) / 2 + lt
        )

    sign = sign.at[0].set((t / jnp.abs(t)) ** ((half_slices[0] - 1) % 2))
    sign = sign.at[1].set((t / jnp.abs(t)) ** ((half_slices[1] - 1) % 2))

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi = cpi.at[0].set(2.0 / jnp.sqrt(2 * el))
    for m in range(2, el + 1):
        cpi = cpi.at[m - 1].set(2.0 / jnp.sqrt(m * (2 * el + 1 - m)))
        cp2 = cp2.at[m - 1].set(cpi[m - 1] / cpi[m - 2])

    # Use Turok & Bucher recursion to evaluate a single half row
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    em = jnp.arange(el+1)

    for i in range(2):
        sgn = (-1) ** (i)

        # Initialise the vector
        dl = dl.at[lims[i]].set(1.0)
        lamb = ((el + 1) * omc - half_slices[i] + c) / s
        dl = dl.at[lims[i] + sgn].set(lamb * dl[lims[i]] * cpi[0])

        for m in range(2, el + 1):
            lamb = ((el + 1) * omc - half_slices[i] + m * c) / s
            dl = dl.at[lims[i] + sgn * m].set(
                lamb * cpi[m - 1] * dl[lims[i] + sgn * (m - 1)]
                - cp2[m - 1] * dl[lims[i] + sgn * (m - 2)]
            )

            dl = lax.cond(dl[lims[i] + sgn * m] > big_const, lambda x: renormalise(x, lrenorm, lims, lbig, bigi, sgn, i, m), lambda x: x, dl)

        # Apply renormalisation
        renorm = sign[i] * jnp.exp(log_first_row[half_slices[i] - 1] - lrenorm[i])

        if i == 0:
            dl = dl.at[lims[i] + sgn * em[:-1]].multiply(renorm)

        if i == 1:
            dl = dl.at[lims[i] + sgn * em].multiply((-1) ** ((mm - em + el) % 2) * renorm)

    return dl

@partial(jit, static_argnums=(2,3,4))
def renormalise(dl, lrenorm, lims, lbig, bigi, sgn, i, m) -> jnp.ndarray:
    lrenorm = lrenorm.at[i].set( lrenorm[i]- lbig)
    for im in range(m + 1):
        dl = dl.at[lims[i] + sgn * im].multiply(bigi)
    return dl

@partial(jit, static_argnums=(1,2))
def north_pole(dl, L, mm) -> jnp.ndarray:
    dl = dl.at[L - 1 + mm].set(1)
    return dl 

@partial(jit, static_argnums=(1,2,3))
def south_pole(dl, L, el, mm) -> jnp.ndarray:
    dl = dl.at[L - 1 - mm].set(-1) ** (el + mm)
    return dl

@partial(jit, static_argnums=(1))
def el0(dl, L) -> jnp.ndarray:
    dl = dl.at[L - 1].set(1)
    return dl
