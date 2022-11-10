import numpy as np
import numpy.fft as fft

import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner


def inverse_sov_fft_healpix(
    flm: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    sampling = "healpix"
    ntheta = samples.ntheta(L, sampling, nside)
    thetas = samples.thetas(L, sampling, nside)

    ftm = np.zeros((ntheta, 4 * nside), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            if el >= np.abs(spin):

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    # See libsharp paper
                    psi_0_y = samples.p2phi_ring(t, 0, nside)
                    m_offset = 1

                    ftm[t, m + L - 1 + m_offset] += (
                        (-1) ** spin * elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                    ) * np.exp(1j * m * psi_0_y)

    f = _healpix_ifft(ftm, ntheta, L, nside)

    return f


def forward_sov_fft_healpix(
    f: np.ndarray, L: int, nside: int, spin: int = 0
) -> np.ndarray:

    sampling = "healpix"
    ntheta = samples.ntheta(L, sampling, nside)
    thetas = samples.thetas(L, sampling, nside)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    ftm = _healpix_fft(f, ntheta, L, nside)

    weights = quadrature.quad_weights_transform(L, sampling, spin=0, nside=nside)
    m_offset = 1

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    # See libsharp paper
                    psi_0_y = samples.p2phi_ring(t, 0, nside)
                    m_offset = 1

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                    ) * np.exp(-1j * m * psi_0_y)

    return flm


def _spectral_folding(ftm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. mitigates aliasing.

    Args:
        ftm (np.ndarray): Partial array of Fourier coefficients for latitude t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Lower resolution set of aliased Fourier coefficients.
    """
    slice_start = L - int(nphi / 2)
    slice_stop = slice_start + nphi
    ftm_slice = ftm[slice_start:slice_stop]

    idx = 1
    while slice_start - idx >= 0:
        ftm_slice[-idx % nphi] += ftm[slice_start - idx]
        idx += 1
    idx = 0
    while slice_stop + idx < len(ftm):
        ftm_slice[idx % nphi] += ftm[slice_stop + idx]
        idx += 1

    return ftm_slice


def _spectral_backprojection(f: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Reflects lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space.

    Args:
        f (np.ndarray): Pixel-space coefficients for ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.
    """
    slice_start = L - int(nphi / 2)
    slice_stop = slice_start + nphi
    ftm = np.zeros(2 * L, dtype=np.complex128)
    ftm[slice_start:slice_stop] = f

    idx = 1
    while slice_start - idx >= 0:
        ftm[slice_start - idx] = f[-idx % nphi]
        idx += 1
    idx = 0
    while slice_stop + idx < len(ftm):
        ftm[slice_stop + idx] = f[idx % nphi]
        idx += 1

    return ftm


def _healpix_fft(f: np.ndarray, ntheta: int, L: int, nside: int) -> np.ndarray:
    """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        ntheta (int): Total number of latitudinal samples (rings).

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    index = 0
    ftm = np.zeros((ntheta, 4 * nside), dtype=np.complex128)
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        ftm_chunk = fft.fftshift(fft.fft(f[index : index + nphi], norm="backward"))
        ftm[t] = (
            ftm_chunk
            if nphi == 4 * nside
            else _spectral_backprojection(ftm_chunk, nphi, L)
        )
        index += nphi
    return ftm


def _healpix_ifft(ftm: np.ndarray, ntheta: int, L: int, nside: int) -> np.ndarray:
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        ntheta (int): Total number of latitudinal samples (rings).

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
    f = np.zeros(samples.f_shape(L, "healpix", nside), dtype=np.complex128)
    index = 0
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        ftm_slice = ftm[t] if nphi == 4 * nside else _spectral_folding(ftm[t], nphi, L)
        f[index : index + nphi] = fft.ifft(fft.ifftshift(ftm_slice), norm="forward")
        index += nphi
    return f
