import numpy as np
import numpy.fft as fft
import s2fft.samples as samples


def periodic_extension(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    r"""Perform period extension of MW/MWSS signal on the sphere in harmonic
    domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.

    Args:
        f (np.ndarray): Signal on the sphere sampled with MW/MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

    Returns:
        np.ndarray: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in same scheme (MW/MWSS) as input.
    """

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
    r"""Perform period extension of MWSS signal on the sphere in spatial domain,
    extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.

    For the MWSS sampling scheme, it is possible to do the period extension in
    :math:`\theta` in the spatial domain.  This is not possible for the MW sampling
    scheme.

    Args:
        f (np.ndarray): Signal on the sphere sampled with MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in MWSS sampling scheme.
    """

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
    r"""Upsample MWSS sampled signal on the sphere defined on domain :math:`[0,\pi]`
    by a factor of two.

    Upsampling is performed by a periodic extension in :math:`\theta` to
    :math:`[0,2\pi]`, followed by zero-padding in harmonic space, followed by
    unextending :math:`\theta` domain back to :math:`[0,\pi]`.

    Args:
        f (np.ndarray): Signal on the sphere sampled with MWSS sampling scheme, sampled
            at resolution L.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere sampled with MWSS sampling scheme, sampling at
        resolution 2*L.
    """

    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext_up = upsample_by_two_mwss_ext(f_ext, L)
    f_up = unextend(f_ext_up, 2 * L, sampling="mwss")

    return f_up


def upsample_by_two_mwss_ext(f_ext: np.ndarray, L: int) -> np.ndarray:
    """Upsample an extended MWSS sampled signal on the sphere defined on domain
    :math:`[0,2\pi]` by a factor of two.

    Upsampling is performed by zero-padding in harmonic space.

    Args:
        f_ext (np.ndarray): Signal on the sphere sampled on extended MWSS sampling
            scheme on domain :math:`[0,2\pi]`, sampled at resolution L.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Signal on the sphere sampled on extended MWSS sampling scheme on
        domain :math:`[0,2\pi]`, sampling at resolution 2*L.
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


def downsample_by_two_mwss(f_ext: np.ndarray, L: int) -> np.ndarray:
    """Downsample an MWSS sampled signal on the sphere.

    Can be applied to either MWSS signal sampled on original domain :math:`[0,\pi]`
    or extended domain :math:`[0,2\pi]`.

    Note:
        Harmonic band-limit of input sampled signal must be even.

    Args:
        f_ext (np.ndarray): Signal on the sphere sampled with MWSS sampling at
            resolution L.

        L (int): Harmonic band-limit.

    Raises:
        ValueError: Harmonic band-limit of input signal must be even.

    Returns:
        np.ndarray: Signal on the sphere sampled with MWSS sampling scheme at
        resolution L/2.  L must be even so that L/2 is an integer.
    """

    # Check L is even.
    if L % 2 != 0:
        raise ValueError(f"L must be even (L={L})")

    f_ext_down = f_ext[0:-1:2, :]

    return f_ext_down


def unextend(f_ext: np.ndarray, L: int, sampling: str = "mw") -> np.ndarray:
    r"""Unextend MW/MWSS sampled signal from :math:`\theta` domain
    :math:`[0,2\pi]` to :math:`[0,\pi]`.

    Args:
        f_ext (np.ndarray): Signal on the sphere sampled on extended :math:`\theta`
            domain :math:`[0,2\pi]`.

        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

        ValueError: Period extension must have correct shape.

    Returns:
        np.ndarray: Signal on the sphere sampled on :math:`\theta` domain
        :math:`[0,\pi]`.
    """

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


def mw_to_mwss_phi(f_mw: np.ndarray, L: int) -> np.ndarray:
    r"""Convert :math:`\phi` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by zero padding in harmonic space.

    Note:
        Can work with arbitrary number of :math:`\theta` samples.  Hence, to convert
        both :math:`(\theta,\phi)` sampling to MWSS, can use :func:`~mw_to_mwss_theta`
        to first convert :math:`\theta` sampling before using this function to convert
        the :math:`\phi` sampling.

    Args:
        f_mw (np.ndarray): Signal on the sphere sampled with MW sampling in
            :math:`\phi` and arbitrary number of samples in

        L (int): Harmonic band-limit.

    Raises:
        ValueError: Input spherical signal must have number of samples in :math:`\phi`
            matching MW sampling.

    Returns:
        np.ndarray: Signal on the sphere with MWSS sampling in :math:`\phi` and
        sampling in :math:`\theta` of the input signal.
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


def mw_to_mwss_theta(f_mw: np.ndarray, L: int, spin: int = 0) -> np.ndarray:
    r"""Convert :math:`\theta` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.

    Args:
        f_mw (np.ndarray): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Raises:
        ValueError: Input spherical signal must have shape matching MW sampling.

    Returns:
        np.ndarray: Signal on the sphere with MWSS sampling in :math:`\theta` and MW
        sampling in :math:`\phi`.
    """

    nphi_mw = samples.nphi_equiang(L, sampling="mw")
    ntheta_mw = samples.ntheta(L, sampling="mw")
    ntheta_mwss = samples.ntheta(L, sampling="mwss")
    ntheta_mw_ext = samples.ntheta_extension(L, sampling="mw")
    ntheta_mwss_ext = samples.ntheta_extension(L, sampling="mwss")

    if f_mw.shape[0] != ntheta_mw:
        raise ValueError(f"Invalid theta size={f_mw.shape[0]} for mw sampling")

    if f_mw.shape[1] != nphi_mw:
        raise ValueError(f"Invalid phi size={f_mw.shape[1]} for mw sampling")

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


def mw_to_mwss(f_mw: np.ndarray, L, spin: int = 0) -> np.ndarray:
    r"""Convert signal on the sphere from MW sampling to MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.  Second, zero padding in harmonic space corresponding to
    :math:`\phi` is performed.

    Args:
        f_mw (np.ndarray): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere sampled with MWSS sampling.
    """
    # Must do extension in this order, i.e. theta first, then phi
    # (since theta extension must do FFTs in phi so shouldn't change number of
    # phi samples there).

    return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)
