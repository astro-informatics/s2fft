import numpy as np
import numpy.fft as fft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner


def inverse_sov_fft_healpix(
    flm: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

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
