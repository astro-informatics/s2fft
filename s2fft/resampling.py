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


def upsample_by_two_mwss(f_ext: np.ndarray, L: int) -> np.ndarray:

    ntheta = samples.ntheta(L, sampling="mwss")
    nphi = samples.nphi_equiang(L, sampling="mwss")
    ntheta_ext = samples.ntheta_extension(L, sampling="mwss")

    # Check shape of f_ext correct
    assert f_ext.shape == (ntheta_ext, nphi)

    # For each phi, perform FFT over 2*pi theta range
    # Put normalisation in forward transform that is at original resolution (otherwise
    # if in backward transform need to manually adjust by multiplying by (4*L) / (2*L).
    f_ext = fft.fftshift(fft.fft(f_ext, axis=0, norm="forward"), axes=0)

    # Zero pad
    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = np.zeros((ntheta_ext_up, nphi), dtype=np.complex128)
    for p in range(0, nphi):
        f_ext_up[L : ntheta_ext + L, p] = f_ext[0:ntheta_ext, p]

    # Perform IFFT to convert back to spatial domain
    f_ext_up = fft.ifft(fft.ifftshift(f_ext_up, axes=0), axis=0, norm="forward")

    return f_ext_up


def downsample_by_two_mwss(f_ext, L):
    """TODO

    Note L is the bandlimit of f_ext, so output is at L/2.
    L must be even.
    """

    # Check L is even.
    if L % 2 != 0:
        raise ValueError(f"L must be even (L={L})")

    f_ext_down = f_ext[0:-1:2, :]

    return f_ext_down
