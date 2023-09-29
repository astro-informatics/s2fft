import numpy as np
from typing import Tuple


def ntheta(L: int = None, sampling: str = "mw", nside: int = None) -> int:
    r"""Number of :math:`\theta` samples for sampling scheme at specified resolution.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: L not specified when sampling not healpix.

        ValueError: HEALPix sampling set but nside not specified.

        ValueError: Sampling scheme not supported.

    Returns:
        int: Number of :math:`\theta` samples of sampling scheme at given resolution.
    """

    if sampling.lower() != "healpix" and L is None:
        raise ValueError(
            f"Sampling scheme sampling={sampling} with L={L} not supported"
        )

    if sampling.lower() == "mw":
        return L

    elif sampling.lower() == "mwss":
        return L + 1

    elif sampling.lower() == "dh":
        return 2 * L

    elif sampling.lower() == "healpix":
        if nside is None:
            raise ValueError(
                f"Sampling scheme sampling={sampling} with nside={nside} not supported"
            )

        return 4 * nside - 1

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def ntheta_extension(L: int, sampling: str = "mw") -> int:
    r"""Number of :math:`\theta` samples for MW/MWSS sampling when extended to
    :math:`2\pi`.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Sampling scheme other than MW/MWSS.

    Returns:
        int: Number of :math:`\theta` samples when extended to :math:`2\pi`.

    """

    if sampling.lower() == "mw":
        return 2 * L - 1

    elif sampling.lower() == "mwss":
        return 2 * L

    else:
        raise ValueError(
            f"Sampling scheme sampling={sampling} does not support periodic extension"
        )


def nphi_equiang(L: int, sampling: str = "mw") -> int:
    r"""Number of :math:`\phi` samples for equiangular sampling scheme at specified
    resolution.

    Number of samples is independent of :math:`\theta` since equiangular sampling
    scheme.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: HEALPix sampling scheme.

        ValueError: Unknown sampling scheme.

    Returns:
        int: Number of :math:`\phi` samples.
    """

    if sampling.lower() == "mw":
        return 2 * L - 1

    elif sampling.lower() == "mwss":
        return 2 * L

    elif sampling.lower() == "dh":
        return 2 * L - 1

    elif sampling.lower() == "healpix":
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    return 1


def ftm_shape(L: int, sampling: str = "mw", nside: int = None) -> Tuple[int, int]:
    r"""Shape of intermediate array, before/after latitudinal step.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.

    Raises:
        ValueError: Unknown sampling scheme.

    Returns:
        Tuple[int,int]: Shape of ftm array, i.e. :math:`[n_{\theta}, n_{\phi}]`. Note
        that here "healpix" defaults to :math:`2L = 4nside` phi samples for ftm.
    """

    if sampling.lower() in ["mwss", "healpix"]:
        return ntheta(L, sampling, nside), 2 * L

    elif sampling.lower() in ["mw", "dh"]:
        return ntheta(L, sampling, nside), 2 * L - 1

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    return 1


def nphi_equitorial_band(nside: int) -> int:
    r"""Number of :math:`\phi` samples within the equitorial band for
    HEALPix sampling scheme.

    Args:
        nside (int, optional): HEALPix Nside resolution parameter.

    Returns:
        int: Number of :math:`\phi` samples.
    """
    return 4 * nside


def nphi_ring(t: int, nside: int = None) -> int:
    r"""Number of :math:`\phi` samples for HEALPix sampling on given :math:`\theta`
    ring.

    Args:
        t (int): Index of HEALPix :math:`\theta` ring.

        nside (int, optional): HEALPix Nside resolution parameter.

    Raises:
        ValueError: Invalid ring index given nside.

    Returns:
        int: Number of :math:`\phi` samples on given :math:`\theta` ring.
    """

    if (t >= 0) and (t < nside - 1):
        return 4 * (t + 1)

    elif (t >= nside - 1) and (t <= 3 * nside - 1):
        return 4 * nside

    elif (t > 3 * nside - 1) and (t <= 4 * nside - 2):
        return 4 * (4 * nside - t - 1)

    else:
        raise ValueError(f"Ring t={t} not contained by nside={nside}")


def thetas(L: int = None, sampling: str = "mw", nside: int = None) -> np.ndarray:
    r"""Compute :math:`\theta` samples for given sampling scheme.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Array of :math:`\theta` samples for given sampling scheme.
    """
    t = np.arange(0, ntheta(L=L, sampling=sampling, nside=nside)).astype(np.float64)

    return t2theta(t, L, sampling, nside)


def t2theta(
    t: int, L: int = None, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    r"""Convert index to :math:`\theta` angle for sampling scheme.

    Args:
        t (int): :math:`\theta` index.

        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: L not specified when sampling not healpix.

        ValueError: HEALPix sampling set but nside not specified.

        ValueError: Sampling scheme not supported.

    Returns:
        np.ndarray: :math:`\theta` angle(s) for passed index or indices.
    """

    if sampling.lower() != "healpix" and L is None:
        raise ValueError(
            f"Sampling scheme sampling={sampling} with L={L} not supported"
        )

    if sampling.lower() == "mw":
        return (2 * t + 1) * np.pi / (2 * L - 1)

    elif sampling.lower() == "mwss":
        return 2 * t * np.pi / (2 * L)

    elif sampling.lower() == "dh":
        return (2 * t + 1) * np.pi / (4 * L)

    elif sampling.lower() == "healpix":
        if nside is None:
            raise ValueError(
                f"Sampling scheme sampling={sampling} with nside={nside} not supported"
            )

        return _t2theta_healpix(t, nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def _t2theta_healpix(t: int, nside: int) -> np.ndarray:
    r"""Convert (ring) index to :math:`\theta` angle for HEALPix sampling scheme.

    Args:
        t (int): :math:`\theta` index.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: :math:`\theta` angle(s) for passed HEALPix (ring) index or indices.
    """

    z = np.zeros_like(t)
    z[t < nside - 1] = 1 - (t[t < nside - 1] + 1) ** 2 / (3 * nside**2)

    z[(t >= nside - 1) & (t <= 3 * nside - 1)] = 4 / 3 - 2 * (
        t[(t >= nside - 1) & (t <= 3 * nside - 1)] + 1
    ) / (3 * nside)

    z[(t > 3 * nside - 1) & (t <= 4 * nside - 2)] = (
        4 * nside - 1 - t[(t > 3 * nside - 1) & (t <= 4 * nside - 2)]
    ) ** 2 / (3 * nside**2) - 1

    return np.arccos(z)


def phis_ring(t: int, nside: int) -> np.ndarray:
    r"""Compute :math:`\phi` samples for given :math:`\theta` HEALPix ring.

    Args:
        t (int): :math:`\theta` index.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: :math:`\phi` angles.
    """

    p = np.arange(0, nphi_ring(t, nside)).astype(np.float64)

    return p2phi_ring(t, p, nside)


def p2phi_ring(t: int, p: int, nside: int) -> np.ndarray:
    r"""Convert index to :math:`\phi` angle for HEALPix for given :math:`\theta` ring.

    Args:
        t (int): :math:`\theta` index of ring.

        p (int): :math:`\phi` index within ring.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: :math:`\phi` angle.
    """

    shift = 1 / 2
    if (t + 1 >= nside) & (t + 1 <= 3 * nside):
        shift *= (t - nside + 2) % 2
        factor = np.pi / (2 * nside)
        return factor * (p + shift)
    elif t + 1 > 3 * nside:
        factor = np.pi / (2 * (4 * nside - t - 1))
    else:
        factor = np.pi / (2 * (t + 1))
    return factor * (p + shift)


def phis_equiang(L: int, sampling: str = "mw") -> np.ndarray:
    r"""Compute :math:`\phi` samples for equiangular sampling scheme.

    Args:
        L (int, optional): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported equiangular sampling
            schemes include {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Array of :math:`\phi` samples for given sampling scheme.
    """
    p = np.arange(0, nphi_equiang(L, sampling))

    return p2phi_equiang(L, p, sampling)


def p2phi_equiang(L: int, p: int, sampling: str = "mw") -> np.ndarray:
    r"""Convert index to :math:`\phi` angle for sampling scheme.

    Args:
        L (int, optional): Harmonic band-limit.

        p (int): :math:`\phi` index.

        sampling (str, optional): Sampling scheme.  Supported equiangular sampling
            schemes include {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: HEALPix sampling not support (only equiangular schemes supported).

        ValueError: Unknown sampling scheme.

    Returns:
        np.ndarray: :math:`\phi` sample(s) for given sampling scheme.
    """

    if sampling.lower() == "mw":
        return 2 * p * np.pi / (2 * L - 1)

    elif sampling.lower() == "mwss":
        return 2 * p * np.pi / (2 * L)

    elif sampling.lower() == "dh":
        return 2 * p * np.pi / (2 * L - 1)

    elif sampling.lower() == "healpix":
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def ring_phase_shift_hp(
    L: int,
    t: int,
    nside: int,
    forward: bool = False,
    reality: bool = False,
) -> np.ndarray:
    r"""Generates a phase shift vector for HEALPix for a given :math:`\theta` ring.

    Args:
        L (int, optional): Harmonic band-limit.

        t (int): :math:`\theta` index of ring.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Returns:
        np.ndarray: Vector of phase shifts with shape :math:`[2L-1]`.
    """
    phi_offset = p2phi_ring(t, 0, nside)
    sign = -1 if forward else 1
    m_start_ind = 0 if reality else -L + 1
    return np.exp(sign * 1j * np.arange(m_start_ind, L) * phi_offset)


def f_shape(L: int = None, sampling: str = "mw", nside: int = None) -> Tuple[int]:
    r"""Shape of spherical signal.

    Args:
        L (int, optional): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        Tuple[int]: Pixel-space array dimensions with shape :math:`[n_{\theta}, n_{\phi}]`.
        Note that "healpix" is instead indexed by a 1D array, with standard conventions.
    """

    if sampling.lower() != "healpix" and L is None:
        raise ValueError(
            f"Sampling scheme sampling={sampling} with L={L} not supported"
        )

    if sampling.lower() == "healpix" and nside is None:
        raise ValueError(
            f"Sampling scheme sampling={sampling} with nside={nside} not supported"
        )

    if sampling.lower() == "healpix":
        return (12 * nside**2,)

    else:
        return ntheta(L, sampling), nphi_equiang(L, sampling)


def flm_shape(L: int) -> Tuple[int, int]:
    r"""Standard shape of harmonic coefficients.

    Args:
        L (int, optional): Harmonic band-limit.

    Returns:
        Tuple[int]: Sampling array shape, with indexing :math:`[\ell, m]`.
    """

    return L, 2 * L - 1


def elm2ind(el: int, m: int) -> int:
    r"""Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.

    1D index is defined by `el**2 + el + m`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

    Returns:
        int: Corresponding 1D index value.
    """

    return el**2 + el + m


def ind2elm(ind: int) -> tuple:
    r"""Convert from 1D spherical harmonic index to 2D index of :math:`(\ell,m)`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        ind (int): 1D spherical harmonic index.

    Returns:
        tuple: `(el,m)` defining spherical harmonic degree and order.
    """

    el = np.floor(np.sqrt(ind))

    m = ind - el**2 - el

    return el, m


def ncoeff(L: int) -> int:
    """Number of spherical harmonic coefficients for given band-limit L.

    Args:
        L (int, optional): Harmonic band-limit.

    Returns:
        int: Number of spherical harmonic coefficients.
    """

    return elm2ind(L - 1, L - 1) + 1


def hp_ang2pix(nside: int, theta: float, phi: float) -> int:
    r"""Convert angles to HEALPix index for HEALPix ring ordering scheme.

    Args:
        nside (int): HEALPix Nside resolution parameter.

        theta (float): Spherical :math:`\theta` angle.

        phi (float): Spherical :math:`\phi` angle.

    Returns:
        int: HEALPix map index for ring ordering scheme.
    """

    z = np.cos(theta)

    return _hp_zphi2pix(nside, z, phi)


def _hp_zphi2pix(nside: int, z: float, phi: float) -> int:
    r"""Convert angles to HEALPix index for HEALPix ring ordering scheme, using
    :math:`z=\cos(\theta)`.

    Note:
        Translated function from HEALPix Java implementation.

    Args:
        nside (int): HEALPix Nside resolution parameter.

        z (float): Cosine of spherical :math:`\theta` angle, i.e. :math:`\cos(\theta)`.

        phi (float): Spherical :math:`\phi` angle.

    Returns:
        int: HEALPix map index for ring ordering scheme.
    """

    tt = 2 * phi / np.pi
    za = np.abs(z)
    nl2 = int(2 * nside)
    nl4 = int(4 * nside)
    ncap = int(nl2 * (nside - 1))
    npix = int(12 * nside**2)
    if za < 2 / 3:  # equatorial region
        jp = int(nside * (0.5 + tt - 0.75 * z))
        jm = int(nside * (0.5 + tt + 0.75 * z))

        ir = int(nside + 1 + jp - jm)
        kshift = 0
        if ir % 2 == 0:
            kshift = 1
        ip = int((jp + jm - nside + kshift + 1) / 2) + 1
        ipix1 = ncap + nl4 * (ir - 1) + ip

    else:  # North and South polar caps
        tp = tt - int(tt)
        tmp = np.sqrt(3.0 * (1.0 - za))
        jp = int(nside * tp * tmp)
        jm = int(nside * (1.0 - tp) * tmp)

        ir = jp + jm + 1
        ip = int(tt * ir) + 1
        if ip > 4 * ir:
            ip = ip - 4 * ir

        ipix1 = 2 * ir * (ir - 1) + ip
        if z <= 0.0:
            ipix1 = npix - 2 * ir * (ir + 1) + ip

    return ipix1 - 1


def hp_getidx(L: int, el: int, m: int) -> int:
    r"""Compute HEALPix harmonic index.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix :math`\ell_{\text{max}}` convention,
        where :math:`L = \ell_{\text{max}} + 1`.

    Args:
        L (int): Harmonic band-limit.

        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

    Returns:
        int: Corresponding index for RING ordered HEALPix.
    """
    return m * (2 * L - 1 - m) // 2 + el


def flm_2d_to_1d(flm_2d: np.ndarray, L: int) -> np.ndarray:
    r"""Convert from 2D indexed harmonic coefficients to 1D indexed coefficients.
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_2d (np.ndarray): 2D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 1D indexed harmonic coefficients.
    """
    flm_1d = np.zeros(ncoeff(L), dtype=np.complex128)

    if len(flm_2d.shape) != 2:
        if len(flm_2d.shape) == 1:
            raise ValueError(f"Flm is already 1D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 1D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_1d[elm2ind(el, m)] = flm_2d[el, L - 1 + m]

    return flm_1d


def flm_1d_to_2d(flm_1d: np.ndarray, L: int) -> np.ndarray:
    r"""Convert from 1D indexed harmnonic coefficients to 2D indexed coefficients.    
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_1d (np.ndarray): 1D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 2D indexed harmonic coefficients.
    """

    flm_2d = np.zeros(flm_shape(L), dtype=np.complex128)

    if len(flm_1d.shape) != 1:
        if len(flm_1d.shape) == 2:
            raise ValueError(f"Flm is already 2D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 2D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_2d[el, L - 1 + m] = flm_1d[elm2ind(el, m)]

    return flm_2d


def flm_hp_to_2d(flm_hp: np.ndarray, L: int) -> np.ndarray:
    r"""Converts from HEALPix (healpy) indexed harmonic coefficients to 2D indexed
    coefficients.
    
    Notes:
        HEALPix implicitly assumes conjugate symmetry and thus only stores positive `m` 
        coefficients. Here we unpack that into harmonic coefficients of an 
        explicitly real signal.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix `lmax` convention,
        where `L = lmax + 1`.

    Note:
        Storage conventions for harmonic coefficients :math:`f_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::
            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{HEALPix}: [flm_{(0,0)}, \dots, flm_{(2,0)}, flm_{(1,1)}, \dots, flm_{(L-1,1)}, \dots]

    Note:
        Returns harmonic coefficients of an explicitly real signal.

    Args:
        flm_hp (np.ndarray): HEALPix indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 2D indexed harmonic coefficients.
    
    """
    flm_2d = np.zeros(flm_shape(L), dtype=np.complex128)

    if len(flm_hp.shape) != 1:
        raise ValueError(f"Healpix indexed flms are not flat")

    for el in range(L):
        flm_2d[el, L - 1 + 0] = flm_hp[hp_getidx(L, el, 0)]
        for m in range(1, el + 1):
            flm_2d[el, L - 1 + m] = flm_hp[hp_getidx(L, el, m)]
            flm_2d[el, L - 1 - m] = (-1) ** m * np.conj(flm_2d[el, L - 1 + m])

    return flm_2d


def flm_2d_to_hp(flm_2d: np.ndarray, L: int) -> np.ndarray:
    r"""Converts from 2D indexed harmonic coefficients to HEALPix (healpy) indexed
    coefficients.
    
    Note:
        HEALPix implicitly assumes conjugate symmetry and thus only stores positive `m` 
        coefficients. So this function discards the negative `m` values. This process 
        is NOT invertible! See the `healpy api docs <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.alm2map.html>`_ 
        for details on healpy indexing and lengths.

    Note:
        Storage conventions for harmonic coefficients :math:`f_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::
            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{HEALPix}: [flm_{(0,0)}, \dots, flm_{(2,0)}, flm_{(1,1)}, \dots, flm_{(L-1,1)}, \dots]

    Warning:
        Returns harmonic coefficients of an explicitly real signal.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix `lmax` convention,
        where `L = lmax + 1`.

    Args:
        flm_2d (np.ndarray): 2D indexed harmonic coefficients.

        L (int): Harmonic band-limit.
        
    Returns:
        np.ndarray: HEALPix indexed harmonic coefficients.
        
    """

    flm_hp = np.zeros(int(L * (L + 1) / 2), dtype=np.complex128)

    if len(flm_hp.shape) != 1:
        raise ValueError(f"HEALPix indexed flms are not flat")

    for el in range(L):
        for m in range(0, el + 1):
            flm_hp[hp_getidx(L, el, m)] = flm_2d[el, L - 1 + m]

    return flm_hp


def lm2lm_hp(flm: np.ndarray, L: int) -> np.ndarray:
    r"""Converts from 1D indexed harmonic coefficients to HEALPix (healpy) indexed
    coefficients.

    Note:
        HEALPix implicitly assumes conjugate symmetry and thus only stores positive `m`
        coefficients. So this function discards the negative `m` values. This process
        is NOT invertible! See the `healpy api docs <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.alm2map.html>`_
        for details on healpy indexing and lengths.

    Note:
        Storage conventions for harmonic coefficients :math:`f_{(\ell,m)}`, for
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

        .. math::

            \text{HEALPix}: [flm_{(0,0)}, \dots, flm_{(2,0)}, flm_{(1,1)}, \dots, flm_{(L-1,1)}, \dots]

    Warning:
        Returns harmonic coefficients of an explicitly real signal.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix :math`\ell_{\text{max}}`
        convention, where :math:`L = \ell_{\text{max}} + 1`.

    Args:
        flm (np.ndarray): 1D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: HEALPix indexed harmonic coefficients.

    """
    flm_hp = np.zeros(int(L * (L + 1) / 2), dtype=np.complex128)

    for el in range(0, L):
        for m in range(0, el + 1):
            flm_hp[hp_getidx(L, el, m)] = flm[elm2ind(el, m)]

    return flm_hp
