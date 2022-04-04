from random import sample
import numpy as np
import numpy.fft as fft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
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

                    for p, phi in enumerate(phis_equiang):

                        f[t, p] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * flm[el, m + L - 1]
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

                    ftm[t, m + L - 1] += (
                        (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * flm[el, m + L - 1]
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

                    m_offset = 1 if sampling == "mwss" else 0
                    ftm[t, m + L - 1 + m_offset] += (
                        (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * flm[el, m + L - 1]
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

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    weights = quadrature.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(phis_equiang):

                        flm[el, m + L - 1] += (
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

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ntheta = samples.ntheta(L, sampling)
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[t, p]

    weights = quadrature.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    flm[el, m + L - 1] += (
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

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                    )

    return flm


def inverse_direct_healpix(
    flm: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    f = np.zeros(12 * nside**2, dtype=np.complex128)

    thetas = samples.thetas(L, "healpix", nside=nside)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(samples.phis_ring(t, nside)):

                        f[samples.hp_ang2pix(nside, theta, phi)] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * flm[el, m + L - 1]
                        )

    return f


def inverse_sov_healpix(
    flm: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, "healpix", nside=nside)

    f = np.zeros(12 * nside**2, dtype=np.complex128)

    thetas = samples.thetas(L, "healpix", nside=nside)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    ftm[t, m + L - 1] += (
                        (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * flm[el, m + L - 1]
                    )

    for t, theta in enumerate(thetas):

        for p, phi in enumerate(samples.phis_ring(t, nside)):

            for m in range(-(L - 1), L):

                f[samples.hp_ang2pix(nside, theta, phi)] += ftm[t, m + L - 1] * np.exp(
                    1j * m * phi
                )

    return f


def inverse_sov_fft_healpix(
    flm: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    raise ValueError(f"Healpix sov + fft not yet functional")

    # TODO: Check flm shape consistent with L
    from scipy.signal import resample

    ntheta = samples.ntheta(L, "healpix", nside)

    f = np.zeros(12 * nside**2, dtype=np.complex128)

    thetas = samples.thetas(L, "healpix", nside)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            # TODO: only need quarter of dl plane here and elsewhere
            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    # See libsharp paper
                    psi_0_y = samples.p2phi_ring(t, 0, nside)

                    ftm[t, m + L - 1] += (
                        (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * flm[el, m + L - 1]
                    ) * np.exp(1j * m * psi_0_y)
    index = 0
    for t, theta in enumerate(thetas):
        nphi = samples.nphi_ring(t, nside)
        f_ring = fft.ifft(fft.ifftshift(ftm[t]), norm="forward")
        f_ring = resample(f_ring, nphi)
        f[index : index + nphi] = f_ring
        index += nphi

    return f


def forward_direct_healpix(
    f: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    # TODO: Check f shape consistent with L

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, "healpix", nside)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    weights = quadrature.quad_weights(L, "healpix", nside=nside)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(samples.phis_ring(t, nside)):

                        flm[el, m + L - 1] += (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * f[samples.hp_ang2pix(nside, theta, phi)]
                        )

    return flm


def forward_sov_healpix(f: np.ndarray, L: int, nside: int, spin: int = 0) -> np.ndarray:

    # TODO: Check f shape consistent with L

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, "healpix", nside)

    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    ntheta = samples.ntheta(L, "healpix", nside)
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(samples.phis_ring(t, nside)):

                ftm[t, m + L - 1] += (
                    np.exp(-1j * m * phi) * f[samples.hp_ang2pix(nside, theta, phi)]
                )

    weights = quadrature.quad_weights(L, "healpix", nside=nside)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1, -spin + L - 1]
                        * ftm[t, m + L - 1]
                    )

    return flm


def inverse_direct_turok(
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

            dl = wigner.turok.compute_full(theta, el, L)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(phis_equiang):

                        f[t, p] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * flm[el, m + L - 1]
                        )

    return f


def inverse_direct_turok_spinslice(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_spin_slice(theta, el, L, spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(phis_equiang):

                        f[t, p] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1]
                            * (-1) ** (-m - spin)
                            * flm[el, m + L - 1]
                        )

    return f
