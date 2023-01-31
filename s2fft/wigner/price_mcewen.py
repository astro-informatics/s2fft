from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial

from s2fft import samples


def generate_precomputes(
    L: int,
    spin: int,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
) -> np.ndarray:

    mm = -spin

    # Correct for mw to mwss conversion
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        beta = samples.thetas(2 * L, "mwss")[1:-1]
    else:
        beta = samples.thetas(L, sampling, nside)

    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L)
    nel = len(el)  # Number of harmonic modes.

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    cs = c / s
    t = np.tan(-beta / 2.0)
    lt = np.log(np.abs(t))
    c2 = np.cos(beta / 2.0)
    omc = 1.0 - c

    # Indexing boundaries
    half_slices = [el + mm + 1, el - mm + 1]

    # Vectors with indexing -L < m < L adopted throughout
    cpi = np.zeros((L + 1, nel), dtype=np.float64)
    cp2 = np.zeros((L + 1, nel), dtype=np.float64)
    log_first_row = np.zeros((2 * L + 1, ntheta, nel), dtype=np.float64)

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

    for k in range(nel):
        cpi[:, k] = np.roll(cpi[:, k], (nel - k - 1), axis=-1)
        cp2[:, k] = np.roll(cp2[:, k], (nel - k - 1), axis=-1)
    # Then evaluate the negative half row and reflect using
    # Wigner-d symmetry relation.

    # Perform precomputations (these can be done offline)
    msign = np.hstack(((-1) ** (abs(np.arange(L - 1))), np.ones(L)))
    lsign = (-1) ** abs(mm + el)
    vsign = np.einsum("m,l->ml", msign, lsign)
    vsign[: L - 1] *= (-1) ** abs(mm + 1 + L)

    lrenorm = np.zeros((2, ntheta, nel), dtype=np.float64)
    lamb = np.zeros((2, ntheta, nel), np.float64)
    for i in range(2):
        for j in range(ntheta):
            lamb[i, j] = ((el + 1) * omc[j] - half_slices[i] + c[j]) / s[j]
            for k in range(nel):
                lamb[i, j, k] -= (nel - k - 1) * cs[j]
                lrenorm[i, j, k] = log_first_row[half_slices[i][k] - 1, j, k]

    indices = np.repeat(np.expand_dims(np.arange(L), 0), ntheta, axis=0)

    return [lrenorm, lamb, vsign, cpi, cp2, cs, indices]


def compute_all_slices(
    beta: np.ndarray, L: int, spin: int, precomps=None
) -> np.ndarray:
    mm = -spin
    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L)
    nel = len(el)  # Number of harmonic modes.

    # Indexing boundaries
    lims = [0, -1]

    dl_test = np.zeros((2 * L - 1, ntheta, nel), dtype=np.float64)
    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = generate_precomputes(
            beta, L, mm
        )
    else:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        lind = L - 1
        sind = lims[i]
        sgn = (-1) ** (i)
        dl_iter = np.ones((2, ntheta, nel), dtype=np.float64)

        dl_iter[1, :, lind:] = np.einsum(
            "l,tl->tl",
            cpi[0, lind:],
            dl_iter[0, :, lind:] * lamb[i, :, lind:],
        )

        dl_test[sind, :, lind:] = (
            dl_iter[0, :, lind:]
            * vsign[sind, lind:]
            * np.exp(lrenorm[i, :, lind:])
        )
        dl_test[sind + sgn, :, lind - 1 :] = (
            dl_iter[1, :, lind - 1 :]
            * vsign[sind + sgn, lind - 1 :]
            * np.exp(lrenorm[i, :, lind - 1 :])
        )

        dl_entry = np.zeros((ntheta, nel), dtype=np.float64)
        for m in range(2, L):
            index = indices >= L - m - 1
            lamb[i, :, np.arange(L)] += cs
            dl_entry = np.where(
                index,
                np.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb[i])
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


@partial(jit, static_argnums=(1, 2))
def compute_all_slices_jax(
    beta: jnp.ndarray, L: int, spin: int, precomps=None
) -> jnp.ndarray:
    mm = -spin
    ntheta = len(beta)
    el = jnp.arange(L)
    nel = len(el)

    # Indexing boundaries
    lims = [0, -1]

    dl_test = jnp.zeros((2 * L - 1, ntheta, nel), dtype=jnp.float64)
    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = generate_precomputes(
            beta, L, mm
        )
    else:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        lind = L - 1
        sind = lims[i]
        sgn = (-1) ** (i)
        dl_iter = jnp.ones((2, ntheta, nel), dtype=jnp.float64)

        dl_iter = dl_iter.at[1, :, lind:].set(
            jnp.einsum(
                "l,tl->tl",
                cpi[0, lind:],
                dl_iter[0, :, lind:] * lamb[i, :, lind:],
            )
        )

        dl_test = dl_test.at[sind, :, lind:].set(
            dl_iter[0, :, lind:]
            * vsign[sind, lind:]
            * jnp.exp(lrenorm[i, :, lind:])
        )

        dl_test = dl_test.at[sind + sgn, :, lind - 1 :].set(
            dl_iter[1, :, lind - 1 :]
            * vsign[sind + sgn, lind - 1 :]
            * jnp.exp(lrenorm[i, :, lind - 1 :])
        )

        dl_entry = jnp.zeros((ntheta, nel), dtype=jnp.float64)

        def pm_recursion_step(m, args):
            dl_test, dl_entry, dl_iter, lamb, lrenorm = args
            index = indices >= L - m - 1
            lamb = lamb.at[i, :, jnp.arange(L)].add(cs)
            dl_entry = jnp.where(
                index,
                jnp.einsum("l,tl->tl", cpi[m - 1], dl_iter[1] * lamb[i])
                - jnp.einsum("l,tl->tl", cp2[m - 1], dl_iter[0]),
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

            dl_iter = dl_iter.at[0].set(
                jnp.where(index, bigi * dl_iter[1], dl_iter[0])
            )
            dl_iter = dl_iter.at[1].set(
                jnp.where(index, bigi * dl_entry, dl_iter[1])
            )
            lrenorm = lrenorm.at[i].set(
                jnp.where(index, lrenorm[i] + lbig, lrenorm[i])
            )
            return dl_test, dl_entry, dl_iter, lamb, lrenorm

        dl_test, dl_entry, dl_iter, lamb, lrenorm = lax.fori_loop(
            2, L, pm_recursion_step, (dl_test, dl_entry, dl_iter, lamb, lrenorm)
        )
    return dl_test
