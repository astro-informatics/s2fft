import numpy as np
import numpy.fft as fft
import s2fft.sampling as samples


def periodic_extension_mw(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # FFT in phi
    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)
    ntheta = samples.ntheta(L, sampling)
    ntheta_ext = 2 * L - 1
    ftm = np.zeros((ntheta_ext, 2 * L - 1), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[t, p]

    # Periodic extension
    for m in range(-(L - 1), L):
        for t in range(L, 2 * L - 1):
            ftm[t, m] = (+1) ** (spin + m) * ftm[2 * L - 2 - t, m]
            # ftm[t, m] = (-1) ** (spin + m) * ftm[2 * L - 2 - t, m]

    # Inverse FFT in phi
    f_ext = np.zeros_like(ftm, dtype=np.complex128)
    for t in range(0, ntheta_ext):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                f_ext[t, p] += np.exp(1j * m * phi) * ftm[t, m] / (2 * L - 1)

    return f_ext


def periodic_extension_mw_2(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    phis_equiang = samples.phis_equiang(L, sampling)
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = 2 * L - 1
    f_ext = np.zeros((ntheta_ext, 2 * L - 1), dtype=np.complex128)

    f_ext[0:ntheta, 0:nphi] = f[0:ntheta, 0:nphi]

    f_ext = fft.fftshift(fft.fft(f_ext, axis=1, norm="backward"), axes=1)

    # for p, phi in enumerate(phis_equiang):
    for m in range(-(L - 1), L):

        for t in range(L, 2 * L - 1):
            f_ext[t, m + L - 1] = (-1) ** m * f_ext[2 * L - 2 - t, m + L - 1]

    f_ext = fft.ifft(fft.ifftshift(f_ext, axes=1), axis=1, norm="backward")

    return f_ext
