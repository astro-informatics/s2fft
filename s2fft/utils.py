from multiprocessing.sharedctypes import Value
import numpy as np
import s2fft.samples as samples


def generate_flm(L: int, spin: int = 0, reality: bool = False) -> np.ndarray:

    ncoeff = samples.ncoeff(L)

    flm = np.zeros(ncoeff, dtype=np.complex128)

    if reality == False:
        flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)
        # For spin signals all flms for el < spin = 0, therfore first spin**2 coefficients = 0
        flm[: spin**2] = 0.0
        return flm
    else:
        for el in range(spin, L):
            flm[samples.elm2ind(el, 0)] = np.random.rand()
            for em in range(1, el + 1):
                flm[samples.elm2ind(el, em)] = np.random.rand() + 1j * np.random.rand()
                flm[samples.elm2ind(el, -em)] = (-1) ** em * np.conj(
                    flm[samples.elm2ind(el, em)]
                )

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
    flm_2d = np.zeros((L, 2 * L - 1), dtype=np.complex128)

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
