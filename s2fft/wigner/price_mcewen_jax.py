from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
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


@partial(jit, static_argnums=(2, 3, 4, 5))
def latitudinal_step_jax(
    flm: jnp.ndarray,
    beta: jnp.ndarray,
    L: int,
    mm: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:

    ntheta = len(beta)  # Number of theta samples
    el = jnp.arange(L)
    nel = len(el)  # Number of harmonic modes.
    ftm = jnp.zeros(samples.ftm_shape(L, sampling), dtype=jnp.complex128)

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = generate_precomputes(
            beta, L, mm
        )
    else:
        lrenorm, lamb, vsign, cpi, cp2, cs, indices = precomps

    for i in range(2):
        if not (reality and i == 0):
            m_offset = 1 if sampling in ["mwss", "healpix"] and i == 0 else 0

            lind = L - 1
            sind = lims[i]
            sgn = (-1) ** (i)
            dl_iter = jnp.ones((2, ntheta, nel), dtype=jnp.float64)

            dl_iter = dl_iter.at[1, :, lind:].set(
                jnp.einsum(
                    "l,tl->tl",
                    cpi[0, lind:],
                    dl_iter[0, :, lind:] * lamb[i, :, lind:],
                    optimize=True,
                )
            )

            # Sum into transform vector
            ftm = ftm.at[:, sind + m_offset].set(
                jnp.nansum(
                    dl_iter[0, :, lind:]
                    * vsign[sind, lind:]
                    * jnp.exp(lrenorm[i, :, lind:])
                    * flm[lind:, sind],
                    axis=-1,
                )
            )

            # Sum into transform vector
            ftm = ftm.at[:, sind + sgn + m_offset].set(
                jnp.nansum(
                    dl_iter[1, :, lind - 1 :]
                    * vsign[sind + sgn, lind - 1 :]
                    * jnp.exp(lrenorm[i, :, lind - 1 :])
                    * flm[lind - 1 :, sind + sgn],
                    axis=-1,
                )
            )
            dl_entry = jnp.zeros((ntheta, nel), dtype=jnp.float64)

            def pm_recursion_step(m, args):
                ftm, dl_entry, dl_iter, lamb, lrenorm = args

                index = indices >= L - m - 1
                lamb = lamb.at[i, :, jnp.arange(nel)].add(cs)

                dl_entry = jnp.where(
                    index,
                    jnp.einsum(
                        "l,tl->tl",
                        cpi[m - 1],
                        dl_iter[1] * lamb[i],
                        optimize=True,
                    )
                    - jnp.einsum(
                        "l,tl->tl", cp2[m - 1], dl_iter[0], optimize=True
                    ),
                    dl_entry,
                )
                dl_entry = dl_entry.at[:, -(m + 1)].set(1)

                # Sum into transform vector
                ftm = ftm.at[:, sind + sgn * m + m_offset].set(
                    jnp.nansum(
                        dl_entry
                        * vsign[sind + sgn * m]
                        * jnp.exp(lrenorm[i])
                        * flm[:, sind + sgn * m],
                        axis=-1,
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
                return ftm, dl_entry, dl_iter, lamb, lrenorm

            ftm, dl_entry, dl_iter, lamb, lrenorm = lax.fori_loop(
                2, L, pm_recursion_step, (ftm, dl_entry, dl_iter, lamb, lrenorm)
            )
    return ftm


def inverse_transform(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling="mw",
    reality: bool = False,
) -> np.ndarray:
    thetas = samples.thetas(L, sampling)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    ftm = np.zeros(samples.ftm_shape(L, sampling), dtype=np.complex128)

    for t, theta in enumerate(thetas):
        for el in range(abs(spin), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            for m in range(-el, el + 1):
                ftm[t, m + L - 1 + m_offset] += (
                    elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                )
    ftm *= (-1) ** spin
    return np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


@partial(jit, static_argnums=(1, 2, 3, 4))
def inverse_transform_new_jax(
    flm: jnp.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    reality: bool = False,
    precomps=None,
) -> jnp.ndarray:

    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # Apply harmonic normalisation
    flm = jnp.einsum(
        "lm,l->lm",
        flm,
        jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)),
        optimize=True,
    )

    # Perform latitudinal wigner-d recursions
    ftm = latitudinal_step_jax(
        flm, thetas, L, -spin, sampling, reality, precomps
    )

    # Remove south pole singularity
    if sampling in ["mw", "mwss"]:
        ftm = ftm.at[-1].set(0)
        ftm = ftm.at[-1, L - 1 + spin + m_offset].set(
            jnp.nansum((-1) ** abs(jnp.arange(L) - spin) * flm[:, L - 1 + spin])
        )

    # Remove north pole singularity
    if sampling == "mwss":
        ftm = ftm.at[0].set(0)
        ftm = ftm.at[0, L - 1 - spin + m_offset].set(
            jnp.nansum(flm[:, L - 1 - spin])
        )

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    if reality:
        return jnp.fft.irfft(
            ftm[:, L - 1 + m_offset :],
            samples.nphi_equiang(L, sampling),
            axis=1,
            norm="forward",
        )
    else:
        ftm = jnp.conj(jnp.fft.ifftshift(ftm, axes=1))
        return jnp.conj(jnp.fft.fft(ftm, axis=1, norm="backward"))


if __name__ == "__main__":
    from s2fft import samples, wigner, transform, utils
    import pyssht as ssht
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import warnings

    warnings.filterwarnings("ignore")

    sampling = "mw"
    L = 8
    spin = 0
    reality = True

    rng = np.random.default_rng(12341234515)
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f_in = np.zeros((ntheta, nphi), dtype=np.complex128)
    for i in range(ntheta):
        for j in range(nphi):
            f_in[i, j] = rng.uniform() + 1j * rng.uniform()

    flm_1d = ssht.forward(f_in, L, spin, Method=sampling.upper())
    flm = samples.flm_1d_to_2d(flm_1d, L)
    flm_jax = jnp.asarray(flm)

    precomps = generate_precomputes(L, spin, sampling)

    f = np.real(
        ssht.inverse(flm_1d, L, spin, Method=sampling.upper(), Reality=reality)
    )
    f_test = np.real(inverse_transform(flm, L, spin, sampling))
    f_test_2 = np.real(
        inverse_transform_new_jax(flm_jax, L, spin, sampling, reality, precomps)
    )

    error = np.log10(np.abs(f - f_test_2))
    mx, mn = np.nanmax(f), np.nanmin(f)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(f, cmap="magma", vmax=mx, vmin=mn)
    ax2.imshow(f_test, cmap="magma", vmax=mx, vmin=mn)
    ax3.imshow(f_test_2, cmap="magma", vmax=mx, vmin=mn)
    im = ax4.imshow(error, cmap="magma")

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    plt.show()
