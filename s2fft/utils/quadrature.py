from types import ModuleType

import array_api_extra as xpx
import numpy as np
import numpy.fft as fft

from s2fft.sampling import s2_samples as samples


def quad_weights_transform(
    L: int, sampling: str = "mwss", spin: int = 0, nside: int = 0
) -> np.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration *to use in transform* for various sampling schemes.

    Quadrature weights to use in transform for MWSS correspond to quadrature weights
    are twice the base resolution, i.e. 2 * L.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mwss", "dh", "gl", "healpix", "cc"}.  Defaults to "mwss".

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        np.ndarray: Quadrature weights *to use in transform* for sampling scheme for
        each :math:`\theta` (weights are identical as :math:`\phi` varies for given
        :math:`\theta`).

    """
    if sampling.lower() == "mwss":
        return (
            quad_weights_mwss_theta_only(2 * L, spin=0)
            * 2
            * np.pi
            / samples.nphi_equiang(L, "mwss")
        )

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    elif sampling.lower() == "cc":
        return quad_weights_cc(L)

    elif sampling.lower() == "f2":
        return quad_weights_f2(L)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def quad_weights(
    L: int = None, sampling: str = "mw", spin: int = 0, nside: int = None
) -> np.ndarray:
    r"""
    Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration for various sampling schemes.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix", "cc", "f2"}.  Defaults to "mw".

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        np.ndarray: Quadrature weights for sampling scheme for each :math:`\theta`
        (weights are identical as :math:`\phi` varies for given :math:`\theta`).

    """
    if sampling.lower() == "mw":
        return quad_weights_mw(L, spin)

    elif sampling.lower() == "mwss":
        return quad_weights_mwss(L, spin)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "gl":
        return quad_weights_gl(L)

    elif sampling.lower() == "cc":
        return quad_weights_cc(L)

    elif sampling.lower() == "f2":
        return quad_weights_f2(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


def quad_weights_hp(nside: int) -> np.ndarray:
    r"""
    Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi`
    integration.

    Note:
        HEALPix weights are identical for all pixels.  Nevertheless, an array of
        weights is returned (with identical values) for consistency of interface
        across other sampling schemes.

    Args:
        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (all weights in array are
        identical).

    """
    npix = 12 * nside**2
    rings = samples.ntheta(sampling="healpix", nside=nside)
    hp_weights = np.zeros(rings, dtype=np.float64)
    hp_weights[:] = 4 * np.pi / npix

    return hp_weights


def quad_weights_gl_theta_only(L: int) -> np.ndarray:
    r"""
    Compute GL quadrature weights for :math:`\theta` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """
    x1, x2 = -1.0, 1.0
    ntheta = samples.ntheta(L, "gl")
    weights = np.zeros(ntheta, dtype=np.float64)

    m = int((L + 1) / 2)
    x1 = 0.5 * (x2 - x1)

    i = np.arange(1, m + 1)
    z = np.cos(np.pi * (i - 0.25) / (L + 0.5))
    z1 = 2.0
    while np.max(np.abs(z - z1)) > 1e-14:
        p1 = 1.0
        p2 = 0.0
        for j in range(1, L + 1):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
        pp = L * (z * p1 - p2) / (z * z - 1.0)
        z1 = z
        z = z1 - p1 / pp

    weights[i - 1] = 2.0 * x1 / ((1.0 - z**2) * pp * pp)
    weights[L + 1 - i - 1] = weights[i - 1]

    return weights


def quad_weights_gl(L: int) -> np.ndarray:
    r"""
    Compute GL quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_gl_theta_only(L) * 2 * np.pi / samples.nphi_equiang(L, "gl")


def quad_weights_dh(L: int) -> np.ndarray:
    r"""
    Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_dh_theta_only(L) * 2 * np.pi / samples.nphi_equiang(L, "dh")


def quad_weights_dh_theta_only(L: int) -> np.ndarray:
    r"""
    Compute DH quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

    Returns:
        float: Weights computed for each :math:`\theta`.

    """
    thetas = samples.thetas(L, sampling="dh")
    w = 0.0
    for k in range(0, L):
        w += np.sin((2 * k + 1) * thetas) / (2 * k + 1)
    w *= 2 / L * np.sin(thetas)
    return w


def quad_weights_mw(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return (
        quad_weights_mw_theta_only(L, spin) * 2 * np.pi / samples.nphi_equiang(L, "mw")
    )


def quad_weights_mwss(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return (
        quad_weights_mwss_theta_only(L, spin)
        * 2
        * np.pi
        / samples.nphi_equiang(L, "mwss")
    )


def quad_weights_mwss_theta_only(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MWSS quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """
    w = np.zeros(2 * L, dtype=np.complex128)
    # Extra negative m, so logically -el-1 <= m <= el.
    for i in range(-(L - 1) + 1, L + 1):
        w[i + L - 1] = mw_weights(i - 1)

    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L)

    q = wr[: L + 1]

    q[1:L] = q[1:L] + (-1) ** spin * wr[-1:L:-1]

    return q


def quad_weights_mw_theta_only(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Compute MW quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    """
    w = np.zeros(2 * L - 1, dtype=np.complex128)
    for i in range(-(L - 1), L):
        w[i + L - 1] = mw_weights(i)

    w *= np.exp(-1j * np.arange(-(L - 1), L) * np.pi / (2 * L - 1))
    wr = np.real(fft.fft(fft.ifftshift(w), norm="backward")) / (2 * L - 1)
    q = wr[:L]

    q[: L - 1] = q[: L - 1] + (-1) ** spin * wr[-1 : L - 1 : -1]

    return q


def mw_weights(m: int) -> float:
    r"""
    Compute MW weights given as a function of index m.

    MW weights are defined by

    .. math::

        w(m^\prime) = \int_0^\pi \text{d} \theta \sin \theta \exp(i m^\prime\theta),

    which can be computed analytically.

    Args:
        m (int): Harmonic weight index.

    Returns:
        float: MW weight.

    """
    if m == 1:
        return 1j * np.pi / 2

    elif m == -1:
        return -1j * np.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0


def _fejer_second_rule_rfft_weights(n_points: int, xp: ModuleType = np):
    r"""
    Compute real-valued FFT of weights for Fejer second quadrature rule.

    Args:
        n_points (int): Number of quadrature points / nodes (including left boundary point).
        xp (module): Array namespace to use for operations. Defaults to NumPy.

    Returns:
        np.ndarray: Real-valued FFT of array of weights for quadrature nodes. Corresponding weigh
        array computed using `irfft` includes zero weight in first element for point at left (-1)
        boundary but excludes zero weight for point at right (1) boundary.

    References:
       Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195–202. https://doi.org/10.1007/s10543-006-0045-4.

    """
    t = xp.arange(1, n_points, 2)
    w0 = -2 / (t * (t - 2))
    w1 = (
        xp.array([-2 / (n_points - 1)])
        if n_points % 2 == 0
        else xp.array([-1 / (n_points - 2)])
    )
    return xp.concatenate([w0, w1])


def quad_weights_cc(L: int, xp: ModuleType = np) -> np.ndarray:
    r"""
    Compute Clenshaw-Curtis quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.
        xp (module): Array namespace to use for operations. Defaults to NumPy.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_cc_theta_only(L, xp) * 2 * xp.pi / samples.nphi_equiang(L, "cc")


def quad_weights_cc_theta_only(L: int, xp: ModuleType = np) -> np.ndarray:
    r"""
    Compute Clenshaw-Curtis quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.
        xp (module): Array namespace to use for operations. Defaults to NumPy.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    References:
       Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195–202. https://doi.org/10.1007/s10543-006-0045-4.

    """
    if L < 1:
        raise ValueError(f"Bandlimit must be at least 1: L = {L}")

    n_theta = samples.ntheta(L, "cc")

    # Computing RFFT below with n=0 will fail so explicitly return
    # weights for single point case
    if n_theta == 1:
        return xp.array([2.0])

    n = n_theta - 1
    w = _fejer_second_rule_rfft_weights(n, xp)
    g = -xp.ones(n // 2 + 1)
    g = xpx.at(g)[n // 2].add(2 * n if n % 2 == 0 else n)
    g /= n**2 - 1 + (n % 2)
    weights = xp.fft.irfft(w + g, n=n)
    return xp.concatenate([weights, weights[:1]])


def quad_weights_f2(L: int, xp: ModuleType = np) -> np.ndarray:
    r"""
    Compute Fejér's second rule quadrature weights for :math:`\theta` and :math:`\phi` integration.

    Args:
        L (int): Harmonic band-limit.
        xp (module): Array namespace to use for operations. Defaults to NumPy.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).

    """
    return quad_weights_f2_theta_only(L, xp) * 2 * xp.pi / samples.nphi_equiang(L, "f2")


def quad_weights_f2_theta_only(L: int, xp: ModuleType = np) -> np.ndarray:
    r"""
    Compute Fejér's second rule quadrature weights for :math:`\theta` integration (only).

    Args:
        L (int): Harmonic band-limit.
        xp (module): Array namespace to use for operations. Defaults to NumPy.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.

    References:
       Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195–202. https://doi.org/10.1007/s10543-006-0045-4.

    """
    if L < 1:
        raise ValueError(f"Bandlimit must be at least 1: L = {L}")

    n_theta = samples.ntheta(L, "f2")
    w = _fejer_second_rule_rfft_weights(n_theta + 1, xp)
    weights = xp.fft.irfft(w, n=n_theta + 1)
    # Weight is zero at left boundary / north pole which we assume is excluded from nodes
    return weights[1:]
