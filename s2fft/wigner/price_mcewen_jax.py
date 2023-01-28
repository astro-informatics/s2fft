import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)
from functools import partial

from s2fft import samples
from s2fft.wigner.price_mcewen import generate_precomputes


@partial(jit, static_argnums=(1, 2))
def compute_slice_jax(
    beta: jnp.ndarray, L: int, mm: int, precomps=None
) -> jnp.ndarray:

    ntheta = len(beta)
    el = jnp.arange(L)
    nel = len(el)

    # Indexing boundaries
    lims = [0, -1]

    dl_test = jnp.zeros((2 * L - 1, ntheta, nel), dtype=jnp.float64)
    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2, cs = generate_precomputes(beta, L, mm)
    else:
        lrenorm, lamb, vsign, cpi, cp2, cs = precomps

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

        for m in range(2, L):
            lind = L - m - 1
            lamb = lamb.at[i, :, jnp.arange(nel)].add(cs)

            dl_entry = jnp.einsum(
                "l,tl->tl",
                cpi[m - 1, lind:],
                dl_iter[1, :, lind:] * lamb[i, :, lind:],
            )
            dl_entry -= jnp.einsum(
                "l,tl->tl", cp2[m - 1, lind:], dl_iter[0, :, lind:]
            )
            dl_entry = dl_entry.at[:, 0].set(1)

            dl_test = dl_test.at[sind + sgn * m, :, lind:].set(
                dl_entry
                * vsign[sind + sgn * m, lind:]
                * jnp.exp(lrenorm[i, :, lind:])
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = jnp.log(abs(dl_entry))

            dl_iter = dl_iter.at[0, :, lind:].set(bigi * dl_iter[1, :, lind:])
            dl_iter = dl_iter.at[1, :, lind:].set(bigi * dl_entry)
            lrenorm = lrenorm.at[i, :, lind:].add(lbig)

    return dl_test


@partial(jit, static_argnums=(2, 3, 4))
def latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    mm: int,
    sampling: str = "mw",
    precomps=None,
) -> jnp.ndarray:

    ntheta = len(beta)  # Number of theta samples
    el = jnp.arange(L)
    nel = len(el)  # Number of harmonic modes.
    ftm = jnp.zeros(samples.ftm_shape(L, sampling), dtype=jnp.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2, cs = generate_precomputes(beta, L, mm)
    else:
        lrenorm, lamb, vsign, cpi, cp2, cs = precomps

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

        # Sum into transform vector
        ftm = ftm.at[:, sind].set(
            jnp.nansum(
                dl_iter[0, :, lind:]
                * vsign[sind, lind:]
                * jnp.exp(lrenorm[i, :, lind:])
                * flm[lind:, sind],
                axis=-1,
            )
        )

        # Sum into transform vector
        ftm = ftm.at[:, sind + sgn].set(
            jnp.nansum(
                dl_iter[1, :, lind - 1 :]
                * vsign[sind + sgn, lind - 1 :]
                * jnp.exp(lrenorm[i, :, lind - 1 :])
                * flm[lind - 1 :, sind + sgn],
                axis=-1,
            )
        )

        for m in range(2, L):
            lind = L - m - 1
            lamb = lamb.at[i, :, np.arange(nel)].add(cs)

            dl_entry = jnp.einsum(
                "l,tl->tl",
                cpi[m - 1, lind:],
                dl_iter[1, :, lind:] * lamb[i, :, lind:],
            )
            dl_entry -= jnp.einsum(
                "l,tl->tl", cp2[m - 1, lind:], dl_iter[0, :, lind:]
            )
            dl_entry = dl_entry.at[:, 0].set(1)

            # Sum into transform vector
            ftm = ftm.at[:, sind + sgn * m].set(
                jnp.nansum(
                    dl_entry
                    * vsign[sind + sgn * m, lind:]
                    * jnp.exp(lrenorm[i, :, lind:])
                    * flm[lind:, sind + sgn * m],
                    axis=-1,
                )
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = jnp.log(abs(dl_entry))

            dl_iter = dl_iter.at[0, :, lind:].set(bigi * dl_iter[1, :, lind:])
            dl_iter = dl_iter.at[1, :, lind:].set(bigi * dl_entry)
            lrenorm = lrenorm.at[i, :, lind:].add(lbig)

    return ftm


def inverse_transform(
    flm: np.ndarray, L: int, spin: int, sampling="mw"
) -> np.ndarray:
    thetas = samples.thetas(L, sampling)
    ftm = np.zeros(samples.ftm_shape(L, sampling), dtype=np.complex128)

    for t, theta in enumerate(thetas):
        for el in range(abs(spin), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            for m in range(-el, el + 1):
                ftm[t, m + L - 1] += (
                    elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                )
    ftm *= (-1) ** spin
    return np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


@partial(jit, static_argnums=(1, 2, 3))
def inverse_transform_new_jax(
    flm: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    precomps=None,
) -> np.ndarray:
    thetas = samples.thetas(L, sampling)
    flm = jnp.einsum(
        "lm,l->lm", flm, jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi))
    )
    ftm = latitudinal_step_jax(flm, thetas, L, -spin, sampling, precomps)

    # Remove pole singularity
    if sampling == "mw":
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin].set(
            jnp.nansum((-1) ** abs(jnp.arange(L) - spin) * flm[:, L - 1 + spin])
        )

    ftm *= (-1) ** spin
    return jnp.fft.ifft(jnp.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


if __name__ == "__main__":
    from s2fft import samples, wigner, transform, utils
    from matplotlib import pyplot as plt
    import warnings

    warnings.filterwarnings("ignore")

    sampling = "mw"
    L = 16
    spin = 0

    rng = np.random.default_rng(12341234515)
    flm = utils.generate_flm(rng, L, 0, spin)
    flm_jax = jnp.asarray(flm)

    precomps = generate_precomputes(samples.thetas(L, sampling), L, -spin)

    f = np.real(transform.inverse(flm, L, spin, sampling))
    f_test = np.real(inverse_transform(flm, L, spin, sampling))
    f_test_2 = np.real(
        inverse_transform_new_jax(flm_jax, L, spin, sampling, precomps)
    )

    mx, mn = np.nanmax(f), np.nanmin(f)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(f, cmap="magma", vmax=mx, vmin=mn)
    ax2.imshow(f_test, cmap="magma", vmax=mx, vmin=mn)
    ax3.imshow(f_test_2, cmap="magma", vmax=mx, vmin=mn)
    plt.show()

# if __name__ == "__main__":
#     import s2fft.samples as samples
#     import s2fft.wigner as wigner
#     from s2fft.wigner.price_mcewen import compute_slice

#     import warnings

#     warnings.filterwarnings("ignore")

#     sampling = "mw"
#     L = 8
#     el = L - 1
#     betas = samples.thetas(L, sampling)
#     # beta_ind = int(L / 2)
#     beta_ind = -2
#     beta = betas[beta_ind]
#     spin = 0

#     precomps = generate_precomputes(samples.thetas(L, sampling), L, -spin)

#     dl_turok = wigner.turok.compute_slice(beta, el, L, -spin)
#     dl_price_mcewen_jax = compute_slice_jax(betas, L, -spin, precomps)[
#         :, beta_ind, el
#     ]

#     print(np.nanmax(np.log10(np.abs(dl_turok - dl_price_mcewen_jax))))
#     from matplotlib import pyplot as plt

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.plot(dl_turok, label="turok")
#     ax1.plot(dl_price_mcewen_jax, label=" test")
#     ax1.legend()
#     ax2.plot(np.log10(np.abs(dl_turok - dl_price_mcewen_jax)))
#     ax2.axhline(y=-14, color="r", linestyle="--")
#     plt.show()
