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
    nphi = f.shape[1]
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


def upsample_by_two_mwss(f: np.ndarray, L: int, spin: int = 0) -> np.ndarray:

    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext_up = upsample_by_two_mwss_ext(f_ext, L)
    f_up = unextend(f_ext_up, 2 * L, sampling="mwss")

    return f_up


def upsample_by_two_mwss_ext(f_ext: np.ndarray, L: int) -> np.ndarray:
    """TODO

    This works with theta range over 2*pi
    """

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


def unextend(f_ext, L, sampling: str = "mw"):

    if sampling.lower() not in ["mw", "mwss"]:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )

    ntheta_ext = samples.ntheta_extension(L, sampling)

    if f_ext.shape[0] != ntheta_ext:
        raise ValueError(
            f"Periodic extension has wrong shape (shape={f_ext.shape}, L={L})"
        )

    if sampling.lower() == "mw":

        f = f_ext[0:L, :]

    elif sampling.lower() == "mwss":

        f = f_ext[0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )

    return f


def mw_to_mwss_phi(f_mw, L):
    """TODO

    Note can work with arbitrary number of theta samples
    """
    ntheta = f_mw.shape[0]
    nphi_mw = samples.nphi_equiang(L, sampling="mw")
    nphi_mwss = samples.nphi_equiang(L, sampling="mwss")

    if f_mw.shape[1] != nphi_mw:
        raise ValueError(f"Invalid phi size={f_mw.shape[1]} for mw sampling")

    # FFT in phi
    # TODO: check ok t overwrite input argmument
    ftm_mw = fft.fftshift(fft.fft(f_mw, axis=1, norm="forward"), axes=1)

    # Zero pad
    f_mwss = np.zeros((ntheta, nphi_mwss), dtype=np.complex128)
    for t in range(0, ntheta):
        f_mwss[t, 1 : nphi_mw + 1] = ftm_mw[t, 0:nphi_mw]

    f_mwss = fft.ifft(fft.ifftshift(f_mwss, axes=1), axis=1, norm="forward")

    return f_mwss


def mw_to_mwss_theta(f_mw, L, spin=0):

    nphi_mw = samples.nphi_equiang(L, sampling="mw")
    ntheta_mw = samples.ntheta(L, sampling="mw")
    ntheta_mwss = samples.ntheta(L, sampling="mwss")
    ntheta_mw_ext = samples.ntheta_extension(L, sampling="mw")
    ntheta_mwss_ext = samples.ntheta_extension(L, sampling="mwss")

    if f_mw.shape[0] != ntheta_mw:
        raise ValueError(f"Invalid theta size={f_mw.shape[0]} for mw sampling")

    if f_mw.shape[1] != nphi_mw:
        raise ValueError(f"Invalid phi size={f_mw.shape[1]} for mw sampling")

    # TODO: add support for spin
    f_mw_ext = periodic_extension(f_mw, L, spin=spin, sampling="mw")

    # FFT in theta
    fmp_mw_ext = fft.fftshift(fft.fft(f_mw_ext, axis=0, norm="forward"), axes=0)

    # Zero pad and apply phase shift to account for MW north pole offset
    fmp_mwss_ext = np.zeros((ntheta_mwss_ext, nphi_mw), dtype=np.complex128)
    for p in range(0, nphi_mw):
        fmp_mw_ext[0:ntheta_mw_ext, p] *= np.exp(
            -1j * np.arange(-(L - 1), L) * np.pi / (2 * L - 1)
        )
        fmp_mwss_ext[1 : ntheta_mw_ext + 1, p] = fmp_mw_ext[0:ntheta_mw_ext, p]

    # Perform IFFT to convert back to spatial domain
    f_mwss_ext = fft.ifft(fft.ifftshift(fmp_mwss_ext, axes=0), axis=0, norm="forward")

    # Unextend
    f_mwss = unextend(f_mwss_ext, L, sampling="mwss")

    return f_mwss


def mw_to_mwss(f_mw, L, spin=0):

    # Must do extension in this order, i.e. theta first, then phi
    # (since theta extension must do FFTs in phi so shouldn't change number of
    # phi samples there).

    return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)
