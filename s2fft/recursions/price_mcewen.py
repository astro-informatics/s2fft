import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial

from s2fft.sampling import s2_samples as samples
from typing import List

import warnings

warnings.filterwarnings("ignore")


def generate_precomputes(
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    L_lower: int = 0,
) -> List[np.ndarray]:
    r"""Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    In practice one could compute these on-the-fly but the memory overhead is
    negligible and well worth the acceleration.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[np.ndarray]: List of precomputed coefficient arrays.

    Note:
        TODO: this function should be optimised.
    """
    mm = -spin
    L0 = L_lower
    # Correct for mw to mwss conversion
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        beta = samples.thetas(2 * L, "mwss")[1:-1]
    else:
        beta = samples.thetas(L, sampling, nside)

    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L0, L)

    # Trigonometric constant adopted throughout
    t = np.tan(-beta / 2.0)
    lt = np.log(np.abs(t))
    c2 = np.cos(beta / 2.0)

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]

    # Vectors with indexing -L < m < L adopted throughout
    cpi = np.zeros((L + 1, L - L0), dtype=np.float64)
    cp2 = np.zeros((L + 1, L - L0), dtype=np.float64)
    log_first_row = np.zeros((2 * L + 1, ntheta, L - L0), dtype=np.float64)

    # Populate vectors for first row
    log_first_row[0] = np.einsum("l,t->tl", 2.0 * el, np.log(np.abs(c2)))

    for i in range(2, L + abs(mm) + 2):
        ratio = (2 * el + 2 - i) / (i - 1)
        for j in range(ntheta):
            log_first_row[i - 1, j] = (
                log_first_row[i - 2, j] + np.log(ratio) / 2 + lt[j]
            )

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi[0] = 2.0 / np.sqrt(2 * el)
    for m in range(2, L + 1):
        cpi[m - 1] = 2.0 / np.sqrt(m * (2 * el + 1 - m))
        cp2[m - 1] = cpi[m - 1] / cpi[m - 2]

    for k in range(L0, L):
        cpi[:, k - L0] = np.roll(cpi[:, k - L0], (L - k - 1), axis=-1)
        cp2[:, k - L0] = np.roll(cp2[:, k - L0], (L - k - 1), axis=-1)
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    # Perform precomputations (these can be done offline)
    msign = np.hstack(((-1) ** (abs(np.arange(L - 1))), np.ones(L)))
    lsign = (-1) ** abs(mm + el)
    vsign = np.einsum("m,l->ml", msign, lsign)
    vsign[: L - 1] *= (-1) ** abs(mm + 1 + L)

    lrenorm = np.zeros((2, ntheta, L - L0), dtype=np.float64)
    for i in range(2):
        for j in range(ntheta):
            for k in range(L0, L):
                lrenorm[i, j, k - L0] = log_first_row[
                    half_slices[i][k - L0] - 1, j, k - L0
                ]

    indices = np.repeat(np.expand_dims(np.arange(L0, L), 0), ntheta, axis=0)
    return [lrenorm, vsign, cpi, cp2, indices]


@partial(jit, static_argnums=(0, 2, 3, 4, 5))
def generate_precomputes_jax(
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    L_lower: int = 0,
    betas: jnp.ndarray = None,
) -> List[jnp.ndarray]:
    r"""Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    In practice one could compute these on-the-fly but the memory overhead is
    negligible and well worth the acceleration. JAX implementation of
    :func:`~generate_precomputes`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        beta (jnp.ndarray): Array of polar angles in radians.

    Returns:
        List[jnp.ndarray]: List of precomputed coefficient arrays.
    """
    mm = -spin
    L0 = L_lower
    # Correct for mw to mwss conversion
    if betas is None:
        if forward and sampling.lower() in ["mw", "mwss"]:
            sampling = "mwss"
            beta = samples.thetas(2 * L, "mwss")[1:-1]
        else:
            beta = samples.thetas(L, sampling, nside)
    else:
        beta = betas

    ntheta = len(beta)  # Number of theta samples
    el = jnp.arange(L0, L)

    # Trigonometric constant adopted throughout
    t = jnp.tan(-beta / 2.0)
    lt = jnp.log(jnp.abs(t))
    c2 = jnp.cos(beta / 2.0)

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]

    # Vectors with indexing -L < m < L adopted throughout
    cpi = jnp.zeros((L + 1, L - L0), dtype=jnp.float64)
    cp2 = jnp.zeros((L + 1, L - L0), dtype=jnp.float64)

    # Initialising coefficients cp(m)= cplus(l-m).
    cpi = cpi.at[0].add(2.0 / jnp.sqrt(2 * el))

    def cpi_cp2_loop(m, args):
        cpi, cp2 = args
        cpi = cpi.at[m - 1].add(2.0 / jnp.sqrt(m * (2 * el + 1 - m)))
        cp2 = cp2.at[m - 1].add(cpi[m - 1] / cpi[m - 2])
        return cpi, cp2

    cpi, cp2 = lax.fori_loop(2, L + 1, cpi_cp2_loop, (cpi, cp2))

    def cpi_cp2_roll_loop(m, args):
        cpi, cp2 = args
        cpi = cpi.at[:, m - L0].set(jnp.roll(cpi[:, m - L0], (L - m - 1), axis=-1))
        cp2 = cp2.at[:, m - L0].set(jnp.roll(cp2[:, m - L0], (L - m - 1), axis=-1))
        return cpi, cp2

    cpi, cp2 = lax.fori_loop(L0, L, cpi_cp2_roll_loop, (cpi, cp2))

    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    # Perform precomputations (these can be done offline)
    msign = jnp.hstack(((-1) ** (abs(jnp.arange(L - 1))), jnp.ones(L)))
    lsign = (-1) ** abs(mm + el)
    vsign = jnp.einsum("m,l->ml", msign, lsign, optimize=True)
    vsign = vsign.at[: L - 1].multiply((-1) ** abs(mm + 1 + L))

    # Populate vectors for first ro
    lrenorm = jnp.zeros((2, ntheta, L - L0), dtype=jnp.float64)
    log_first_row_iter = jnp.einsum(
        "l,t->tl", 2.0 * el, jnp.log(jnp.abs(c2)), optimize=True
    )

    ratio_update = jnp.arange(2 * L + 1)
    ratio = jnp.repeat(jnp.expand_dims(2 * el + 2, -1), 2 * L + 1, axis=-1)
    ratio -= ratio_update
    ratio /= ratio_update - 1
    ratio = jnp.log(jnp.swapaxes(ratio, 0, 1)) / 2

    for ind in range(2):
        lrenorm = lrenorm.at[ind].set(
            jnp.where(1 == half_slices[ind], log_first_row_iter, lrenorm[ind])
        )

    def renorm_m_loop(i, args):
        log_first_row_iter, lrenorm = args
        log_first_row_iter += ratio[i]
        log_first_row_iter = jnp.swapaxes(log_first_row_iter, 0, 1)
        log_first_row_iter += lt
        log_first_row_iter = jnp.swapaxes(log_first_row_iter, 0, 1)
        for ind in range(2):
            lrenorm = lrenorm.at[ind].set(
                jnp.where(i == half_slices[ind], log_first_row_iter, lrenorm[ind])
            )
        return log_first_row_iter, lrenorm

    _, lrenorm = lax.fori_loop(
        2, L + abs(mm) + 2, renorm_m_loop, (log_first_row_iter, lrenorm)
    )

    indices = jnp.repeat(jnp.expand_dims(jnp.arange(L0, L), 0), ntheta, axis=0)

    # Remove redundant nans:
    # - in forward pass these are not accessed, so are irrelevant.
    # - in backward pass the adjoint computation otherwise accumulates these
    #   nans into grads if not explicitly set to zero.
    lrenorm = jnp.nan_to_num(lrenorm, nan=0.0, posinf=0.0, neginf=0.0)
    cpi = jnp.nan_to_num(cpi, nan=0.0, posinf=0.0, neginf=0.0)
    cp2 = jnp.nan_to_num(cp2, nan=0.0, posinf=0.0, neginf=0.0)

    return [lrenorm, vsign, cpi, cp2, indices]


def generate_precomputes_wigner(
    L: int,
    N: int,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    L_lower: int = 0,
) -> List[List[np.ndarray]]:
    r"""Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    In practice one could compute these on-the-fly but the memory overhead is
    negligible and well worth the acceleration. This is a wrapped extension of
    :func:`~generate_precomputes` for the case of multiple spins, i.e. the Wigner
    transform over SO(3).

    Args:
        L (int): Harmonic band-limit.

        N (int): Azimuthal bandlimit

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[List[np.ndarray]]: 2N-1 length List of Lists of precomputed coefficient arrays.

    Note:
        TODO: this function should be optimised.
    """
    precomps = []
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps.append(generate_precomputes(L, -n, sampling, nside, forward, L_lower))
    return precomps


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def generate_precomputes_wigner_jax(
    L: int,
    N: int,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    L_lower: int = 0,
) -> List[List[jnp.ndarray]]:
    r"""Compute recursion coefficients with :math:`\mathcal{O}(L^2)` memory overhead.
    In practice one could compute these on-the-fly but the memory overhead is
    negligible and well worth the acceleration. This is a wrapped extension of
    :func:`~generate_precomputes` for the case of multiple spins, i.e. the Wigner
    transform over SO(3). JAX implementation of :func:`~generate_precomputes_wigner`.

    Args:
        L (int): Harmonic band-limit.

        N (int): Azimuthal bandlimit

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        List[List[jnp.ndarray]]: 2N-1 length List of Lists of precomputed coefficient arrays.
    """
    lrenorm = []
    vsign = []
    cpi = []
    cp2 = []
    indices = []
    captured_repeats = False
    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        precomps = generate_precomputes_jax(L, -n, sampling, nside, forward, L_lower)
        lrenorm.append(precomps[0])
        vsign.append(precomps[1])
        if not captured_repeats:
            cpi.append(precomps[2])
            cp2.append(precomps[3])
            indices.append(precomps[4])
            captured_repeats = True

    return [
        jnp.asarray(lrenorm),
        jnp.asarray(vsign),
        jnp.asarray(cpi),
        jnp.asarray(cp2),
        jnp.asarray(indices),
    ]


def compute_all_slices(
    beta: np.ndarray, L: int, spin: int, precomps=None
) -> np.ndarray:
    r"""Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix for all sampled polar angles
    :math:`\beta` and all :math:`\ell` using Price & McEwen recursion.

    The Wigner-d slice for all :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`. The Price & McEwen recursion is analytically correct
    from :math:`-\ell < m < \ell` however numerically it can become unstable for
    :math:`m > 0`. To avoid this we compute :math:`d_{m,
    m^{\prime}}^{\ell}(\beta)` for negative :math:`m` and then evaluate
    :math:`d_{m, -m^{\prime}}^{\ell}(\beta) = (-1)^{m-m^{\prime}} d_{-m,
    m^{\prime}}^{\ell}(\beta)` which we can again evaluate using the same recursion.

    On-the-fly renormalisation is implemented to avoid potential over/under-flows,
    within any given iteration of the recursion the iterants are :math:`\sim \mathcal{O}(1)`.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 - m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires :math:`\sim 2L`
    operations.

    Args:
        beta (np.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray]): Precomputed recursion coefficients with memory overhead
            :math:`\mathcal{O}(L^2)`, which is minimal.

    Returns:
        np.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.
    """
    # Indexing boundaries and constants
    mm = -spin
    ntheta = len(beta)
    lims = [0, -1]
    el = np.arange(L)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    omc = 1.0 - c

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]

    dl_test = np.zeros((2 * L - 1, ntheta, L), dtype=np.float64)
    if precomps is None:
        lrenorm, offset, vsign, cpi, cp2, cs, indices = generate_precomputes(
            beta, L, mm
        )
    else:
        lrenorm, offset, vsign, cpi, cp2, cs, indices = precomps

    lamb = np.zeros((ntheta, L), np.float64)
    for i in range(2):
        lind = L - 1
        sind = lims[i]
        sgn = (-1) ** (i)
        dl_iter = np.ones((2, ntheta, L), dtype=np.float64)

        lamb = (
            np.einsum("l,t->tl", el + 1, omc)
            + np.einsum("l,t->tl", 2 - L + el, c)
            - half_slices[i]
        )
        lamb = np.einsum("tl,t->tl", lamb, 1 / s)
        dl_iter[1, :, lind:] = np.einsum(
            "l,tl->tl",
            cpi[0, lind:],
            dl_iter[0, :, lind:] * lamb[:, lind:],
        )

        dl_test[sind, :, lind:] = (
            dl_iter[0, :, lind:] * vsign[sind, lind:] * np.exp(lrenorm[i, :, lind:])
        )
        dl_test[sind + sgn, :, lind - 1 :] = (
            dl_iter[1, :, lind - 1 :]
            * vsign[sind + sgn, lind - 1 :]
            * np.exp(lrenorm[i, :, lind - 1 :])
        )

        dl_entry = np.zeros((ntheta, L), dtype=np.float64)
        for m in range(2, L):
            index = indices >= L - m - 1

            lamb = (
                np.einsum("l,t->tl", el + 1, omc)
                + np.einsum("l,t->tl", m - L + el + 1, c)
                - half_slices[i]
            )
            lamb = np.einsum("tl,t->tl", lamb, 1 / s)

            dl_entry = np.where(
                index,
                np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb)
                - np.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
                dl_entry,
            )
            dl_entry[:, -(m + 1)] = 1

            dl_test[sind + sgn * m] = np.where(
                index,
                dl_entry * vsign[sind + sgn * m] * np.exp(lrenorm[i]),
                dl_test[sind + sgn * m],
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = np.log(abs(dl_entry))

            dl_iter[0] = np.where(index, bigi * dl_iter[1], dl_iter[0])
            dl_iter[1] = np.where(index, bigi * dl_entry, dl_iter[1])
            lrenorm[i] = np.where(index, lrenorm[i] + lbig, lrenorm[i])

    return dl_test


@partial(jit, static_argnums=(1, 3, 4, 5))
def compute_all_slices_jax(
    beta: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    forward: bool = False,
    nside: int = None,
    precomps=None,
) -> jnp.ndarray:
    r"""Compute a particular slice :math:`m^{\prime}`, denoted `mm`,
    of the complete Wigner-d matrix for all sampled polar angles
    :math:`\beta` and all :math:`\ell` using Price & McEwen recursion.

    The Wigner-d slice for all :math:`\ell` (`el`) and :math:`\beta` is
    computed recursively over :math:`m` labelled 'm' at a specific
    :math:`m^{\prime}`. The Price & McEwen recursion is analytically correct
    from :math:`-\ell < m < \ell` however numerically it can become unstable for
    :math:`m > 0`. To avoid this we compute :math:`d_{m,
    m^{\prime}}^{\ell}(\beta)` for negative :math:`m` and then evaluate
    :math:`d_{m, -m^{\prime}}^{\ell}(\beta) = (-1)^{m-m^{\prime}} d_{-m,
    m^{\prime}}^{\ell}(\beta)` which we can again evaluate using the same recursion.

    On-the-fly renormalisation is implemented to avoid potential over/under-flows,
    within any given iteration of the recursion the iterants are :math:`\sim \mathcal{O}(1)`.

    The Wigner-d slice :math:`d^\ell_{m, m^{\prime}}(\beta)` is indexed for
    :math:`-L < m < L` by `dl[L - 1 - m, \beta, \ell]`. This implementation has
    computational scaling :math:`\mathcal{O}(L)` and typically requires :math:`\sim 2L`
    operations.

    Args:
        beta (jnp.ndarray): Array of polar angles in radians.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        precomps (List[np.ndarray]): Precomputed recursion coefficients with memory overhead
            :math:`\mathcal{O}(L^2)`, which is minimal.

    Returns:
        jnp.ndarray: Wigner-d matrix mm slice of dimension :math:`[2L-1, n_{\theta}, n_{\ell}]`.
    """
    # Indexing boundaries and constants
    mm = -spin
    ntheta = len(beta)
    lims = [0, -1]

    # Trigonometric constant adopted throughout
    c = jnp.cos(beta)
    s = jnp.sin(beta)
    omc = 1.0 - c
    el = jnp.arange(L)

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]

    dl_test = jnp.zeros((2 * L - 1, ntheta, L), dtype=jnp.float64)
    if precomps is None:
        lrenorm, vsign, cpi, cp2, indices = generate_precomputes_jax(
            L, spin, sampling, nside, forward, 0, beta
        )
    else:
        lrenorm, vsign, cpi, cp2, indices = precomps

    for i in range(2):
        lind = L - 1
        sind = lims[i]
        sgn = (-1) ** (i)
        dl_iter = jnp.ones((2, ntheta, L), dtype=jnp.float64)

        lamb = (
            jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
            + jnp.einsum("l,t->tl", 2 - L + el, c, optimize=True)
            - half_slices[i]
        )
        lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

        dl_iter = dl_iter.at[1, :, lind:].set(
            jnp.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[:, lind:],
            )
        )

        dl_test = dl_test.at[sind, :, lind:].set(
            dl_iter[0, :, lind:] * vsign[sind, lind:] * jnp.exp(lrenorm[i, :, lind:])
        )

        dl_test = dl_test.at[sind + sgn, :, lind - 1 :].set(
            dl_iter[1, :, lind - 1 :]
            * vsign[sind + sgn, lind - 1 :]
            * jnp.exp(lrenorm[i, :, lind - 1 :])
        )

        dl_entry = jnp.zeros((ntheta, L), dtype=jnp.float64)

        def pm_recursion_step(m, args):
            dl_test, dl_entry, dl_iter, lrenorm, indices, omc, c, s = args
            index = indices >= L - m - 1

            lamb = (
                jnp.einsum("l,t->tl", el + 1, omc, optimize=True)
                + jnp.einsum("l,t->tl", m - L + el + 1, c, optimize=True)
                - half_slices[i]
            )
            lamb = jnp.einsum("tl,t->tl", lamb, 1 / s, optimize=True)

            dl_entry = jnp.where(
                index,
                jnp.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb, optimize=True)
                - jnp.einsum("l,tl->tl", cp2[m - 1], dl_iter[0], optimize=True),
                dl_entry,
            )
            dl_entry = dl_entry.at[:, -(m + 1)].set(1)

            dl_test = dl_test.at[sind + sgn * m].set(
                jnp.where(
                    index,
                    dl_entry * vsign[sind + sgn * m] * jnp.exp(lrenorm[i]),
                    dl_test[sind + sgn * m],
                )
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = jnp.log(abs(dl_entry))

            dl_iter = dl_iter.at[0].set(jnp.where(index, bigi * dl_iter[1], dl_iter[0]))
            dl_iter = dl_iter.at[1].set(jnp.where(index, bigi * dl_entry, dl_iter[1]))
            lrenorm = lrenorm.at[i].set(jnp.where(index, lrenorm[i] + lbig, lrenorm[i]))
            return dl_test, dl_entry, dl_iter, lrenorm, indices, omc, c, s

        dl_test, dl_entry, dl_iter, lrenorm, indices, omc, c, s = lax.fori_loop(
            2,
            L,
            pm_recursion_step,
            (dl_test, dl_entry, dl_iter, lrenorm, indices, omc, c, s),
        )
    return dl_test
