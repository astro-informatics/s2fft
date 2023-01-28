import numpy as np
from s2fft import samples


def generate_precomputes(beta: np.ndarray, L: int, mm: int) -> np.ndarray:
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
    vsign[: L - 1] *= (-1) ** abs(mm + 1)

    lrenorm = np.zeros((2, ntheta, nel), dtype=np.float64)
    lamb = np.zeros((2, ntheta, nel), np.float64)
    for i in range(2):
        for j in range(ntheta):
            lamb[i, j] = ((el + 1) * omc[j] - half_slices[i] + c[j]) / s[j]
            for k in range(nel):
                lamb[i, j, k] -= (nel - k - 1) * cs[j]
                lrenorm[i, j, k] = log_first_row[half_slices[i][k] - 1, j, k]

    return [lrenorm, lamb, vsign, cpi, cp2]


def compute_vectorised_slice(beta: np.ndarray, L: int, mm: int) -> np.ndarray:
    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L)
    nel = len(el)  # Number of harmonic modes.

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    cs = c / s

    # Indexing boundaries
    lims = [0, -1]

    dl_test = np.zeros((2 * L - 1, ntheta, nel), dtype=np.float64)
    lrenorm, lamb, vsign, cpi, cp2 = generate_precomputes(beta, L, mm)

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

        for m in range(2, L):
            lind = L - m - 1
            lamb[i, :, np.arange(nel)] += cs

            dl_entry = np.einsum(
                "l,tl->tl",
                cpi[m - 1, lind:],
                dl_iter[1, :, lind:] * lamb[i, :, lind:],
            )
            dl_entry -= np.einsum(
                "l,tl->tl", cp2[m - 1, lind:], dl_iter[0, :, lind:]
            )
            dl_entry[:, 0] = 1
            dl_test[sind + sgn * m, :, lind:] = (
                dl_entry
                * vsign[sind + sgn * m, lind:]
                * np.exp(lrenorm[i, :, lind:])
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = np.log(abs(dl_entry))

            dl_iter[0, :, lind:] = bigi * dl_iter[1, :, lind:]
            dl_iter[1, :, lind:] = bigi * dl_entry
            lrenorm[i, :, lind:] += lbig

    return dl_test


def latitudinal_step(
    flm: np.ndarray,
    beta: np.ndarray,
    L: int,
    mm: int,
    sampling: str = "mw",
    precomps=None,
) -> np.ndarray:

    ntheta = len(beta)  # Number of theta samples
    el = np.arange(L)
    nel = len(el)  # Number of harmonic modes.
    ftm = np.zeros(samples.ftm_shape(L, sampling), dtype=np.complex128)

    # Trigonometric constant adopted throughout
    c = np.cos(beta)
    s = np.sin(beta)
    cs = c / s

    # Indexing boundaries
    lims = [0, -1]

    if precomps is None:
        lrenorm, lamb, vsign, cpi, cp2 = generate_precomputes(beta, L, mm)
    else:
        lrenorm, lamb, vsign, cpi, cp2 = precomps

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

        # Sum into transform vector
        ftm[:, sind] = np.nansum(
            dl_iter[0, :, lind:]
            * vsign[sind, lind:]
            * np.exp(lrenorm[i, :, lind:])
            * flm[lind:, sind],
            axis=-1,
        )

        # Sum into transform vector
        ftm[:, sind + sgn] = np.nansum(
            dl_iter[1, :, lind - 1 :]
            * vsign[sind + sgn, lind - 1 :]
            * np.exp(lrenorm[i, :, lind - 1 :])
            * flm[lind - 1 :, sind + sgn],
            axis=-1,
        )

        for m in range(2, L):
            lind = L - m - 1
            lamb[i, :, np.arange(nel)] += cs

            dl_entry = np.einsum(
                "l,tl->tl",
                cpi[m - 1, lind:],
                dl_iter[1, :, lind:] * lamb[i, :, lind:],
            )
            dl_entry -= np.einsum(
                "l,tl->tl", cp2[m - 1, lind:], dl_iter[0, :, lind:]
            )
            dl_entry[:, 0] = 1

            # Sum into transform vector
            ftm[:, sind + sgn * m] = np.nansum(
                dl_entry
                * vsign[sind + sgn * m, lind:]
                * np.exp(lrenorm[i, :, lind:])
                * flm[lind:, sind + sgn * m],
                axis=-1,
            )

            bigi = 1.0 / abs(dl_entry)
            lbig = np.log(abs(dl_entry))

            dl_iter[0, :, lind:] = bigi * dl_iter[1, :, lind:]
            dl_iter[1, :, lind:] = bigi * dl_entry
            lrenorm[i, :, lind:] += lbig

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


def inverse_transform_new(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str = "mw",
    precomps=None,
) -> np.ndarray:
    thetas = samples.thetas(L, sampling)
    flm = np.einsum(
        "lm,l->lm", flm, np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    )
    ftm = latitudinal_step(flm, thetas, L, -spin, sampling, precomps)

    # Remove pole singularity
    if sampling == "mw":
        ftm[-1] = 0
        ftm[-1, L - 1 + spin] = np.nansum(
            (-1) ** abs(np.arange(L) - spin) * flm[:, L - 1 + spin]
        )

    ftm *= (-1) ** spin
    return np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


if __name__ == "__main__":
    from s2fft import samples, wigner, transform, utils
    from matplotlib import pyplot as plt
    import warnings
    import time

    warnings.filterwarnings("ignore")

    sampling = "mw"
    L = 128
    spin = -3

    rng = np.random.default_rng(12341234515)
    flm = utils.generate_flm(rng, L, 0, spin)

    precomps = generate_precomputes(samples.thetas(L, sampling), L, -spin)

    f = np.real(transform.inverse(flm, L, spin, sampling))
    f_test = np.real(inverse_transform(flm, L, spin, sampling))
    f_test_2 = np.real(inverse_transform_new(flm, L, spin, sampling, precomps))

    mx, mn = np.nanmax(f), np.nanmin(f)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(f, cmap="magma", vmax=mx, vmin=mn)
    ax2.imshow(f_test, cmap="magma", vmax=mx, vmin=mn)
    ax3.imshow(f_test_2, cmap="magma", vmax=mx, vmin=mn)
    plt.show()

# if __name__ == "__main__":
#     import s2fft.samples as samples
#     import s2fft.wigner as wigner
#     import warnings

#     warnings.filterwarnings("ignore")

#     sampling = "mw"
#     L = 32
#     el = L - 1
#     betas = samples.thetas(L, sampling)
#     beta_ind = int(L / 2)
#     beta = betas[beta_ind]
#     spin = 2

#     dl_turok = wigner.turok.compute_slice(beta, el, L, -spin)
#     dl_price_mcewen = compute_vectorised_slice(betas, L, -spin)[:, beta_ind, el]

#     print(np.nanmax(np.log10(np.abs(dl_turok - dl_price_mcewen))))
#     from matplotlib import pyplot as plt

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.plot(dl_turok, label="turok")
#     ax1.plot(dl_price_mcewen, label=" test")
#     ax1.legend()
#     ax2.plot(np.log10(np.abs(dl_turok - dl_price_mcewen)))
#     ax2.axhline(y=-14, color="r", linestyle="--")
#     plt.show()
