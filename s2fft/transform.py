import numpy as np
import s2fft.sampling as samples
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

    for t, theta in enumerate(thetas):

        weight = samples.weight_dh(theta, L) * 2 * np.pi / (2 * L - 1)

        for el in range(0, L):

            dl = wigner.risbo.compute_full(dl, theta, L, el)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    i = samples.elm2ind(el, m)

                    for p, phi in enumerate(phis_equiang):

                        flm[i] += (
                            weight
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1, -spin + L - 1]
                            * f[t, p]
                        )

    return flm
