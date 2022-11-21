import numpy as np
from typing import Tuple

# TODO: Extend to other sampling schemes, currently "healpix" not supported.
def f_shape(L: int, N: int, sampling: str = "mw") -> Tuple[int, int, int]:
    r"""Computes the pixel-space sampling shape for rotation group :math:`SO(3)`.

    Importantly, the convention we are using is the :math:`yzz` euler convention, i.e. :math:`[\beta, \alpha, \gamma]` which is to simplify indexing for internal FFTs.

    Args:
        L (int): Harmonic band-limit.

        N (int): Number of Fourier coefficients for tangent plane rotations (i.e. directionality).

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

    Raises:
        ValueError: Unknown sampling scheme.

    Returns:
        (Tuple[int,int,int]): Shape of pixel-space sampling of rotation group :math:`SO(3)`.
    """
    if sampling in ["mw", "mwss", "dh"]:

        return _nbeta(L, sampling), _nalpha(L, sampling), _ngamma(N)

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def flmn_shape(L: int, N: int) -> Tuple[int, int, int]:
    r"""Computes the shape of Wigner coefficients for rotation group :math:`SO(3)`.

    TODO: Add support for compact storage etc.

    Args:
        L (int): Harmonic band-limit.

        N (int): Number of Fourier coefficients for tangent plane rotations (i.e. directionality).

    Returns:
        (Tuple[int,int,int]): Shape of Wigner space sampling of rotation group :math:`SO(3)`.
    """
    return L, 2 * L - 1, 2 * N - 1


def flmn_n_size(L: int, N: int) -> int:
    r"""Computes the number of non-zero Wigner coefficients.

    TODO: Add support for compact storage etc.

    Args:
        L (int): Harmonic band-limit.

        N (int): Number of Fourier coefficients for tangent plane rotations (i.e. directionality).

    Returns:
        (int): Total number of non-zero Wigner coefficients.
    """
    return (2 * N - 1) * L * L


def _nalpha(L: int, sampling: str = "mw") -> int:
    r"""Computes the number of :math:`\alpha` samples.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

    Raises:
        ValueError: Unknown sampling scheme.

    Returns:
        (int): Number of :math:`\alpha` samples.
    """
    if sampling.lower() in ["mw", "dh"]:

        return 2 * L - 1

    elif sampling.lower() == "mwss":

        return 2 * L

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    return 1


def _nbeta(L: int, sampling: str = "mw") -> int:
    r"""Computes the number of :math:`\beta` samples.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

    Raises:
        ValueError: Unknown sampling scheme.

    Returns:
        (int): Number of :math:`\beta` samples.
    """
    if sampling.lower() == "mw":

        return L

    elif sampling.lower() == "mwss":

        return L + 1

    elif sampling.lower() == "dh":

        return 2 * L

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    return 1


def _ngamma(N: int) -> int:
    r"""Computes the number of :math:`\gamma` samples.

    Args:
        N (int): Number of Fourier coefficients for tangent plane rotations (i.e. directionality).

    Returns:
        (int): Number of :math:`\gamma` samples, by default :math:`2N-1`.
    """
    return 2 * N - 1


def elm2ind(el: int, m: int, n: int, L: int, N: int = 1) -> int:
    """Convert from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.

    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

        n (int): Directional order :math:`n`.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations (i.e. directionality). Defaults to 1.

    Returns:
        (int): Corresponding 1D index in Wigner space.
    """
    n_offset = (N - 1 + n) * L * L
    el_offset = el * el
    return n_offset + el_offset + el + m


def flmn_3d_to_1d(flmn_3d: np.ndarray, L: int, N: int = 1) -> np.ndarray:
    r"""Convert from 3D indexed Wigner coefficients to 1D indexed coefficients.

    Args:
        flm_3d (np.ndarray): 3D indexed Wigner coefficients, index order :math:`[\el, m, n]`.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations (i.e. directionality). Defaults to 1.

    Raises:
        ValueError: Flmn is already 1D indexed.

        ValueError: Flmn is not 3D.

    Returns:
        np.ndarray: 1D indexed Wigner coefficients, C flatten index priority :math:`n, \el, m`.
    """
    flmn_1d = np.zeros(flmn_n_size(L, N), dtype=np.complex128)

    if len(flmn_3d.shape) != 3:
        if len(flmn_3d.shape) == 1:
            raise ValueError(f"Flmn is already 1D indexed")
        else:
            raise ValueError(
                f"Cannot convert flmn of dimension {flm_3d.shape} to 1D indexing"
            )
    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_1d[elm2ind(el, m, n, L, N)] = flmn_3d[el, L - 1 + m, N - 1 + n]

    return flmn_1d


def flmn_1d_to_3d(flmn_1d: np.ndarray, L: int, N: int = 1) -> np.ndarray:
    r"""Convert from 1D indexed Wigner coefficients to 3D indexed coefficients.

    Args:
        flm_1d (np.ndarray): 1D indexed Wigner coefficients, C flatten index priority :math:`n, \el, m`.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations (i.e. directionality). Defaults to 1.

    Raises:
        ValueError: Flmn is already 3D indexed.

        ValueError: Flmn is not 1D.

    Returns:
        np.ndarray: 3D indexed Wigner coefficients, index order :math:`[\el, m, n]`.
    """
    flmn_3d = np.zeros(flmn_shape(L, N), dtype=np.complex128)

    if len(flmn_1d.shape) != 1:
        if len(flmn_1d.shape) == 3:
            raise ValueError(f"Flmn is already 3D indexed")
        else:
            raise ValueError(
                f"Cannot convert flmn of dimension {flm_1d.shape} to 3D indexing"
            )
    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_3d[el, L - 1 + m, N - 1 + n] = flmn_1d[elm2ind(el, m, n, L, N)]

    return flmn_3d
