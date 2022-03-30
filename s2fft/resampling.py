import numpy as np
import numpy.fft as fft
import s2fft.sampling as samples


def periodic_extension(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = samples.ntheta_extension(L, sampling)
    f_ext = np.zeros((ntheta_ext, nphi), dtype=np.complex128)

    f_ext[0:ntheta, 0:nphi] = f[0:ntheta, 0:nphi]

    f_ext = fft.fftshift(fft.fft(f_ext, axis=1, norm="backward"), axes=1)

    m_offset = 1 if sampling == "mwss" else 0
    for m in range(-(L - 1), L):

        for t in range(L + m_offset, 2 * L - 1 + m_offset):
            f_ext[t, m + L - 1 + m_offset] = (-1) ** (m + spin) * f_ext[
                2 * L - 2 - t + 2 * m_offset, m + L - 1 + m_offset
            ]

    f_ext = fft.ifft(fft.ifftshift(f_ext, axes=1), axis=1, norm="backward")

    return f_ext


def periodic_extension_spatial_mwss(f: np.ndarray, L: int, spin: int = 0) -> np.ndarray:

    ntheta = samples.ntheta(L, sampling="mwss")
    nphi = samples.nphi_equiang(L, sampling="mwss")
    ntheta_ext = samples.ntheta_extension(L, sampling="mwss")
    f_ext = np.zeros((ntheta_ext, nphi), dtype=np.complex128)

    # Copy samples over sphere, i.e. 0 <= theta <= pi
    f_ext[0:ntheta, 0:nphi] = f[0:ntheta, 0:nphi]

    # Reflect about north pole and add pi shift in phi
    f_ext[ntheta:, 0 : 2 * L] = (-1) ** spin * np.fft.fftshift(
        np.flipud(f[1 : ntheta - 1, 0 : 2 * L]), axes=1
    )

    return f_ext
