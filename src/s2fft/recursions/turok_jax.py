from functools import partial

import jax.lax as lax
import jax.numpy as jnp
from jax import jit

from s2fft.recursions.price_mcewen import (
    _polar_terms,
    _recursion_step,
    _scaled_start_values,
)


@partial(jit, static_argnums=(2, 3))
def compute_slice(beta: float, el: int, L: int, mm: int) -> jnp.ndarray:
    r"""
    Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok &
    Bucher recursion.

    The Wigner-d slice for a given :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`, by two half recursions running inwards from the edges
    :math:`m = \mp\ell`. Each half recursion is only stable while the slice grows in
    its direction, so the halves meet at
    :math:`m^\star = \mathrm{round}(m^\prime\cos\beta)`, the centre of the
    oscillatory region of the slice, rather than at :math:`m = 0`; near the poles the
    slice peaks at :math:`m = \pm m^\prime` and decays away from it. The recursion
    is that of :func:`~s2fft.recursions.price_mcewen.compute_all_slices_jax` for a
    single :math:`\ell`.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 + m]`, and is zero for :math:`|m| > \ell`. This
    implementation has computational scaling :math:`\mathcal{O}(L)` and typically
    requires :math:`\sim 2(L + |m^\prime|)` operations.

    Args:
        beta (float): Polar angle in radians.

        el (int): Harmonic degree of Wigner-d matrix.

        L (int): Harmonic band-limit.

        mm (int): Harmonic order at which to slice the matrix.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension [2L-1].

    """
    beta = jnp.reshape(jnp.asarray(beta, dtype=jnp.float64), (1,))
    el = jnp.reshape(el, (1,))
    start = _scaled_start_values(beta, el, mm, jnp)
    polar = _polar_terms(beta, mm, jnp)

    dl = jnp.zeros(2 * L - 1, dtype=jnp.float64)
    for i in range(2):

        def recursion_step(j, args):
            dl, state = args
            value, state = _recursion_step(i, j, state, start, el, L, mm, polar, jnp)
            m = j - (L - 1) if i == 0 else L - 1 - j
            dl = dl.at[L - 1 + m].add(value[0, 0])
            return dl, state

        state = tuple(jnp.zeros((1, 1), dtype=jnp.float64) for _ in range(3))
        dl, _ = lax.fori_loop(
            0, min(L + abs(mm), 2 * L - 1), recursion_step, (dl, state)
        )

    # Analytically evaluate singularities
    m = jnp.arange(-(L - 1), L)
    valid = el[0] >= abs(mm)
    north_pole = jnp.where((m == mm) & valid, 1.0, 0.0)
    south_pole = jnp.where((m == -mm) & valid, 1.0 - 2.0 * ((el[0] + mm) % 2), 0.0)
    dl = jnp.where(jnp.abs(beta[0]) < 1e-10, north_pole, dl)
    return jnp.where(jnp.abs(beta[0] - jnp.pi) < 1e-10, south_pole, dl)


@partial(jit, static_argnums=(2, 3))
def reindex(dl, el, L, mm) -> jnp.ndarray:
    r"""
    Reorders indexing of Wigner-d matrix.

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
