import numpy as np
import numpy.fft as fft
import s2fft.sampling as samples
import s2fft.resampling as resampling
import s2fft.wigner as wigner


def inverse_direct(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    for p, phi in enumerate(phis_equiang):

                        f[t, p] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * flm[i]
                        )

    return f


def inverse_sov(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    ftm[t, m + L - 1] += (
                        (-1) ** spin * elfactor * dl[m + L - 1, -spin + L - 1] * flm[i]
                    )

    for t, theta in enumerate(thetas):

        for p, phi in enumerate(phis_equiang):

            for m in range(-(L - 1), L):

                f[t, p] += ftm[t, m + L - 1] * np.exp(1j * m * phi)

    return f


def inverse_sov_fft(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    nphi = samples.nphi_equiang(L, sampling)
    ftm = np.zeros((ntheta, nphi), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            # TODO: only need quarter of dl plane here and elsewhere
            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    m_offset = 1 if sampling == "mwss" else 0
                    ftm[t, m + L - 1 + m_offset] += (
                        (-1) ** spin * elfactor * dl[m + L - 1, -spin + L - 1] * flm[i]
                    )

    f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def forward_direct(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check f shape consistent with L

    if sampling.lower() != "dh":

        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    ncoeff = samples.ncoeff(L)

    flm = np.zeros(ncoeff, dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    weights = samples.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    for p, phi in enumerate(phis_equiang):

                        flm[i] += (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * f[t, p]
                        )

    return flm


def forward_sov(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check f shape consistent with L

    if sampling.lower() != "dh":

        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    ncoeff = samples.ncoeff(L)

    flm = np.zeros(ncoeff, dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ntheta = samples.ntheta(L, sampling)
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[t, p]

    weights = samples.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    flm[i] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * ftm[t, m + L - 1]
                    )

    return flm


def forward_sov_fft(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check f shape consistent with L

    if sampling.lower() != "dh":

        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    ncoeff = samples.ncoeff(L)

    flm = np.zeros(ncoeff, dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ntheta = samples.ntheta(L, sampling)
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)

    ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    weights = samples.quad_weights(L, sampling)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    flm[i] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * ftm[t, m + L - 1]
                    )

    return flm


def forward_sov_fft_mwss(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check f shape consistent with L

    # if sampling.lower() not in ["mw", "mwss"]:
    #     raise ValueError(
    #         "Only mw and mwss supported for periodic extension "
    #         f"(not sampling={sampling})"
    #     )

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    ncoeff = samples.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)

    # phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    # ntheta = samples.ntheta(2 * L, sampling)

    ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = samples.quad_weights_transform(L, sampling, spin=0)

    m_offset = 1 if sampling == "mwss" else 0

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    flm[i] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                    )

    return flm
