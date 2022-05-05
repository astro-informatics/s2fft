import numpy as np


def ntheta(L: int = None, sampling: str = "mw", nside: int = None) -> int:
    r"""Number of :math:`\theta` samples for sampling scheme at specified resolution.

    Args:
        L (int): Harmonic band-limit.  Required if sampling not healpix.  Defaults to
            None.

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
        int: Number of :math:`\theta` samples at given resolution when extended to
            :math:`2\pi`.

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

    t = np.arange(0, ntheta(L=L, sampling=sampling, nside=nside)).astype(np.float64)

    return t2theta(t, L, sampling, nside)


def t2theta(
    t: int, L: int = None, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    r"""Convert index to :math:`\theta' angle.

    Args:
        L (int): _description_
        t (int): _description_
        sampling (str, optional): _description_. Defaults to "mw".
        nside (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
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

        # TODO: This should be a separate function.
        z = np.zeros_like(t)
        z[t < nside - 1] = 1 - (t[t < nside - 1] + 1) ** 2 / (3 * nside**2)
        z[(t >= nside - 1) & (t <= 3 * nside - 1)] = 4 / 3 - 2 * (
            t[(t >= nside - 1) & (t <= 3 * nside - 1)] + 1
        ) / (3 * nside)
        z[(t > 3 * nside - 1) & (t <= 4 * nside - 2)] = (
            4 * nside - 1 - t[(t > 3 * nside - 1) & (t <= 4 * nside - 2)]
        ) ** 2 / (3 * nside**2) - 1
        return np.arccos(z)

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def phis_ring(t: int, nside: int) -> np.ndarray:

    p = np.arange(0, nphi_ring(t, nside)).astype(np.float64)

    return p2phi_ring(t, p, nside)


def p2phi_ring(t: int, p: int, nside: int) -> np.ndarray:

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

    p = np.arange(0, nphi_equiang(L, sampling))

    return p2phi_equiang(L, p, sampling)


def p2phi_equiang(L: int, p: int, sampling: str = "mw") -> np.ndarray:

    if sampling.lower() == "mw":

        return 2 * p * np.pi / (2 * L - 1)

    elif sampling.lower() == "mwss":

        return 2 * p * np.pi / (2 * L)

    elif sampling.lower() == "dh":

        return 2 * p * np.pi / (2 * L - 1)

    elif sampling.lower() == "healpix":

        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def flm_shape(L: int) -> tuple:

    return L, 2 * L - 1


def elm2ind(el: int, m: int) -> int:

    return el**2 + el + m


def ind2elm(ind):

    el = np.floor(np.sqrt(ind))

    m = ind - el**2 - el

    return (el, m)


def ncoeff(L):

    return elm2ind(L - 1, L - 1) + 1


def hp_ang2pix(nside: int, theta: float, phi: float) -> int:
    """Translated function from healpix java implementation"""
    z = np.cos(theta)
    return zphi2pix(nside, z, phi)


def zphi2pix(nside: int, z: float, phi: float) -> int:
    """Translated function from healpix java implementation"""
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
    """Returns healpix flm index for l=el & m=em"""
    # return m * (2 * lmax + 1 - m) // 2 + el
    return m * (2 * L - 1 - m) // 2 + el


def flm_2d_to_1d(flm_2d: np.ndarray, L: int) -> np.ndarray:
    r"""Converts from 2d indexed flms to 1d indexed
    
    Conventions for e.g. :math:`L = 3` 

    .. math::

        2D = \begin{bmatrix}
                flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}  \\
                0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                0 & 0 & flm_{(0,0)} & 0 & 0
            \end{bmatrix}
    
    .. math::

        1D =  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Returns:

        1D indexed flms
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
    r"""Converts from 1d indexed flms to 2d indexed
    
    Conventions for e.g. :math:`L = 3` 

    .. math::

        2D = \begin{bmatrix}
                flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}  \\
                0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                0 & 0 & flm_{(0,0)} & 0 & 0
            \end{bmatrix}
    
    .. math::

        1D =  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Returns:

        2D indexed flms
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
    r"""Converts from healpix indexed flms to 2d indexed
    
    Note that healpix implicitly assumes conjugate symmetry and thus 
    only stores positive m coefficients. Here we unpack that into 
    harmonic coefficients of an explicitly real signal.

    Conventions for e.g. :math:`L = 3` 

    .. math::

        2D = \begin{bmatrix}
                flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}  \\
                0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                0 & 0 & flm_{(0,0)} & 0 & 0
            \end{bmatrix}
    
    .. math::

        healpix =  [flm_{0,0}, \dots, flm_{L,0}, flm_{1,1}, \dots, flm_{L,1}, \dots]

    Returns:

        2D indexed flms of a explicitly real signal
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
    r"""Converts from 2d indexed flms to healpix indexed flms
    
    Note that healpix implicitly assumes conjugate symmetry and thus 
    only stores positive m coefficients. So this function discards the 
    negative m values. This process is NOT invertible! See the 
    `healpy api docs <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.alm2map.html>`_ 
    for details on healpix indexing and lengths.

    Conventions for e.g. :math:`L = 3` 

    .. math::

        2D = \begin{bmatrix}
                flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}  \\
                0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                0 & 0 & flm_{(0,0)} & 0 & 0
            \end{bmatrix}
    
    .. math::

        healpix =  [flm_{0,0}, \dots, flm_{L,0}, flm_{1,1}, \dots, flm_{L,1}, \dots]

    Returns:

        healpix indexed flms of an implicitly real signal
    """

    # mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1
    # (L-1) * (2 * (L-1) + 1 - (L-1)) / 2 + (L-1) + 1 = (L-1) * L / 2 + L = L * (L+1)/2
    flm_hp = np.zeros(int(L * (L + 1) / 2), dtype=np.complex128)

    if len(flm_hp.shape) != 1:
        raise ValueError(f"Healpix indexed flms are not flat")

    for el in range(L):
        for m in range(0, el + 1):
            flm_hp[hp_getidx(L, el, m)] = flm_2d[el, L - 1 + m]

    return flm_hp


def lm2lm_hp(flm: np.ndarray, L: int) -> np.ndarray:
    flm_hp = np.zeros(int(L * (L + 1) / 2), dtype=np.complex128)

    for el in range(0, L):
        for m in range(0, el + 1):
            flm_hp[hp_getidx(L, el, m)] = flm[elm2ind(el, m)]

    return flm_hp
