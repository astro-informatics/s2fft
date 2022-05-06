import numpy as np
import numpy.fft as fft
import s2fft.samples as samples


def quad_weight_dh_theta_only(theta: float, L: int) -> float:

    w = 0.0
    for k in range(0, L):
        w += np.sin((2 * k + 1) * theta) / (2 * k + 1)

    w *= 2 / L * np.sin(theta)

    return w


def quad_weights_transform(L: int, sampling: str, spin: int = 0) -> np.ndarray:

    if sampling.lower() == "mwss":
        return quad_weights_mwss_theta_only(2 * L, spin=0) * 2 * np.pi / (2 * L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def quad_weights(L: int, sampling: str, spin: int = 0, nside: int = None) -> np.ndarray:

    if sampling.lower() == "mw":
        return quad_weights_mw(L, spin)

    elif sampling.lower() == "mwss":
        return quad_weights_mwss(L, spin)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


def quad_weights_hp(nside: int) -> np.ndarray:
    npix = 12 * nside**2
    rings = samples.ntheta(0, "healpix", nside)
    hp_weights = np.zeros(rings, dtype=np.float64)
    hp_weights[:] = 4 * np.pi / npix
    return hp_weights


def quad_weights_dh(L):

    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * np.pi / (2 * L - 1)


def quad_weights_mw(L, spin=0):

    return quad_weights_mw_theta_only(L, spin) * 2 * np.pi / (2 * L - 1)


def quad_weights_mwss(L, spin=0):

    return quad_weights_mwss_theta_only(L, spin) * 2 * np.pi / (2 * L)


def quad_weights_mwss_theta_only(L, spin=0):

    w = np.zeros(2 * L, dtype=np.complex128)
    # Extra negative m, so logically -el-1 <= m <= el.
    for i in range(-(L - 1) + 1, L + 1):
        w[i + L - 1] = mw_weights(i - 1)

    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L)

    q = wr[: L + 1]

    q[1:L] = q[1:L] + (-1) ** spin * wr[-1:L:-1]

    return q


def quad_weights_mw_theta_only(L, spin=0):
    """_summary_

    Args:
        L (_type_): _description_
        spin (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    w = np.zeros(2 * L - 1, dtype=np.complex128)
    for i in range(-(L - 1), L):
        w[i + L - 1] = mw_weights(i)

    w *= np.exp(-1j * np.arange(-(L - 1), L) * np.pi / (2 * L - 1))
    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]

    q[: L - 1] = q[: L - 1] + (-1) ** spin * wr[-1 : L - 1 : -1]

    return q


def mw_weights(m):

    if m == 1:
        return 1j * np.pi / 2

    elif m == -1:
        return -1j * np.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0
