import numpy as np
import s2fft.samples as samples


def generate_flm(L: int, spin: int = 0, reality: bool = False) -> np.ndarray:

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    for el in range(spin, L):

        if reality:
            flm[el, 0 + L - 1] = np.random.rand()
        else:
            flm[el, 0 + L - 1] = np.random.rand() + 1j * np.random.rand()

        for m in range(1, el + 1):
            flm[el, m + L - 1] = np.random.rand() + 1j * np.random.rand()
            if reality:
                flm[el, -m + L - 1] = (-1) ** m * np.conj(flm[el, m + L - 1])
            else:
                flm[el, -m + L - 1] = np.random.rand() + 1j * np.random.rand()

    return flm


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
    flm_1d = np.zeros(samples.ncoeff(L), dtype=np.complex128)

    if len(flm_2d.shape) != 2:
        if len(flm_2d.shape) == 1:
            raise ValueError(f"Flm is already 1D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 1D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_1d[samples.elm2ind(el, m)] = flm_2d[el, L - 1 + m]

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
    flm_2d = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if len(flm_1d.shape) != 1:
        if len(flm_1d.shape) == 2:
            raise ValueError(f"Flm is already 2D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 2D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_2d[el, L - 1 + m] = flm_1d[samples.elm2ind(el, m)]

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
    flm_2d = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if len(flm_hp.shape) != 1:
        raise ValueError(f"Healpix indexed flms are not flat")

    for el in range(L):
        flm_2d[el, L - 1 + 0] = flm_hp[samples.hp_getidx(L, el, 0)]
        for m in range(1, el + 1):
            flm_2d[el, L - 1 + m] = flm_hp[samples.hp_getidx(L, el, m)]
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
            flm_hp[samples.hp_getidx(L, el, m)] = flm_2d[el, L - 1 + m]

    return flm_hp


def lm2lm_hp(flm: np.ndarray, L: int) -> np.ndarray:
    flm_hp = np.zeros(int(L * (L + 1) / 2), dtype=np.complex128)

    for el in range(0, L):
        for m in range(0, el + 1):
            flm_hp[samples.hp_getidx(L, el, m)] = flm[samples.elm2ind(el, m)]

    return flm_hp
