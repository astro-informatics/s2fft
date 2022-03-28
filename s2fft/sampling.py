import numpy as np


def ntheta(L: int, sampling: str = "mw") -> int:

    if sampling.lower() == "mw":

        return L

    elif sampling.lower() == "mwss":

        return L + 1

    elif sampling.lower() == "healpix":

        raise NotImplementedError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def nphi_eqiang(L: int, sampling: str = "mw") -> int:

    if sampling.lower() == "mw":

        return 2 * L - 1

    elif sampling.lower() == "mwss":

        return 2 * L

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

    elif sampling.lower() == "healpix":

        raise NotImplementedError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def phis_eqiang(L: int, sampling: str = "mw") -> np.ndarray:

    p = np.arange(0, nphi_eqiang(L, sampling))

    return p2phi_eqiang(L, p, sampling)


def p2phi_eqiang(L: int, p: int, sampling: str = "mw") -> np.ndarray:

    if sampling.lower() == "mw":

        return 2 * p * np.pi / (2 * L - 1)

    elif sampling.lower() == "mwss":

        return 2 * p * np.pi / (2 * L)

    elif sampling.lower() == "healpix":

        raise ValueError(f"Sampling scheme sampling={sampling} not implement")

    else:

        raise ValueError(f"Sampling scheme sampling={sampling} not supported")
