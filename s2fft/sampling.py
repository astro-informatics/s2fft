from tkinter import E
import numpy as np
import numpy.fft as fft


def ntheta(L: int, sampling: str = "mw", nside: int = None) -> int:

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

    if sampling.lower() == "mw":

        return 2 * L - 1

    elif sampling.lower() == "mwss":

        return 2 * L

    else:

        raise ValueError(
            f"Sampling scheme sampling={sampling} does not support periodic extension"
        )


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


def nphi_ring(t: int, nside: int = None) -> int:

    if (t >= 0) and (t < nside - 1):
        return 4 * (t + 1)

    elif (t >= nside - 1) and (t <= 3 * nside - 1):
        return 4 * nside

    elif (t > 3 * nside - 1) and (t <= 4 * nside - 2):
        return 4 * (4 * nside - t - 1)

    else:
        raise ValueError(f"Ring t={t} not contained by nside={nside}")


def thetas(L: int, sampling: str = "mw", nside: int = None) -> np.ndarray:

    t = np.arange(0, ntheta(L, sampling, nside=nside)).astype(np.float64)

    return t2theta(L, t, sampling, nside)


def t2theta(L: int, t: int, sampling: str = "mw", nside: int = None) -> np.ndarray:

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
    rings = ntheta(0, "healpix", nside)
    hp_weights = np.zeros(rings, dtype=np.float64)
    hp_weights[:] = np.pi / (3 * nside**2)
    return hp_weights


def quad_weights_dh(L):

    q = quad_weight_dh_theta_only(thetas(L, sampling="dh"), L)

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
