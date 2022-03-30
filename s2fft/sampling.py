import numpy as np
import numpy.fft as fft


def ntheta(L: int, sampling: str = "mw") -> int:

    if sampling.lower() == "mw":

        return L

    elif sampling.lower() == "mwss":

        return L + 1

    elif sampling.lower() == "dh":

        return 2 * L

    elif sampling.lower() == "healpix":

        raise NotImplementedError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def nphi_equiang(L: int, sampling: str = "mw") -> int:

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


def thetas(L: int, sampling: str = "mw") -> np.ndarray:

    t = np.arange(0, ntheta(L, sampling))

    return t2theta(L, t, sampling)


def t2theta(L: int, t: int, sampling: str = "mw") -> np.ndarray:

    if sampling.lower() == "mw":

        return (2 * t + 1) * np.pi / (2 * L - 1)

    elif sampling.lower() == "mwss":

        return 2 * t * np.pi / (2 * L)

    elif sampling.lower() == "dh":

        return (2 * t + 1) * np.pi / (4 * L)

    elif sampling.lower() == "healpix":

        raise NotImplementedError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


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

        raise ValueError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def elm2ind(el: int, m: int) -> int:

    return el**2 + el + m


def ind2elm(ind):

    el = np.floor(np.sqrt(ind))

    m = ind - el**2 - el

    return (el, m)


def ncoeff(L):

    return elm2ind(L - 1, L - 1) + 1


def quad_weight_dh_theta_only(theta: float, L: int) -> float:

    w = 0.0
    for k in range(0, L):
        w += np.sin((2 * k + 1) * theta) / (2 * k + 1)

    w *= 2 / L * np.sin(theta)

    return w


def quad_weights(L: int, sampling: str) -> np.ndarray:

    if sampling.lower() == "mw":
        return quad_weights_mw(L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implement")


def quad_weights_dh(L):

    q = quad_weight_dh_theta_only(thetas(L, sampling="dh"), L)

    return q * 2 * np.pi / (2 * L - 1)


def quad_weights_mw(L):

    return quad_weights_mw_theta_only(L) * 2 * np.pi / (2 * L - 1)


def quad_weights_mw_theta_only(L):

    w = np.zeros(2 * L - 1, dtype=np.complex)
    for i in range(-(L - 1), L):
        w[i + L - 1] = mw_weights(i)

    w *= np.exp(-1j * np.arange(-(L - 1), L) * np.pi / (2 * L - 1))
    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]

    q[: L - 1] = q[: L - 1] + wr[-1 : L - 1 : -1]

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
