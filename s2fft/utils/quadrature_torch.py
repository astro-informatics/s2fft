import torch
from s2fft.sampling import s2_samples as samples


def quad_weights_transform(
    L: int, sampling: str = "mwss", nside: int = 0
) -> torch.tensor:
    r"""Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration *to use in transform* for various sampling schemes. Torch implementation of
    :func:`~s2fft.quadrature.quad_weights_transform`.

    Quadrature weights to use in transform for MWSS correspond to quadrature weights
    are twice the base resolution, i.e. 2 * L.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mwss", "dh", "healpix}.  Defaults to "mwss".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        torch.tensor: Quadrature weights *to use in transform* for sampling scheme for
        each :math:`\theta` (weights are identical as :math:`\phi` varies for given
        :math:`\theta`).
    """

    if sampling.lower() == "mwss":
        return quad_weights_mwss_theta_only(2 * L) * 2 * torch.pi / (2 * L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")


def quad_weights(
    L: int = None, sampling: str = "mw", nside: int = None
) -> torch.tensor:
    r"""Compute quadrature weights for :math:`\theta` and :math:`\phi`
    integration for various sampling schemes. Torch implementation of
    :func:`~s2fft.quadrature.quad_weights`.

    Args:
        L (int, optional): Harmonic band-limit.  Required if sampling not healpix.
            Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Invalid sampling scheme.

    Returns:
        torch.tensor: Quadrature weights for sampling scheme for each :math:`\theta`
        (weights are identical as :math:`\phi` varies for given :math:`\theta`).
    """

    if sampling.lower() == "mw":
        return quad_weights_mw(L)

    elif sampling.lower() == "mwss":
        return quad_weights_mwss(L)

    elif sampling.lower() == "dh":
        return quad_weights_dh(L)

    elif sampling.lower() == "healpix":
        return quad_weights_hp(nside)

    else:
        raise ValueError(f"Sampling scheme sampling={sampling} not implemented")


def quad_weights_hp(nside: int) -> torch.tensor:
    r"""Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi`
    integration. Torch implementation of :func:`s2fft.quadrature.quad_weights_hp`.

    Note:
        HEALPix weights are identical for all pixels.  Nevertheless, an array of
        weights is returned (with identical values) for consistency of interface
        across other sampling schemes.

    Args:
        nside (int): HEALPix Nside resolution parameter.

    Returns:
        torch.tensor: Weights computed for each :math:`\theta` (all weights in array are
        identical).
    """
    npix = 12 * nside**2
    rings = samples.ntheta(sampling="healpix", nside=nside)
    return torch.ones(rings, dtype=torch.float64) * 4 * torch.pi / npix


def quad_weights_dh(L: int) -> torch.tensor:
    r"""Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.
    Torch implementation of :func:`s2fft.quadrature.quad_weights_dh`.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        torch.tensor: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    q = quad_weight_dh_theta_only(samples.thetas(L, sampling="dh"), L)

    return q * 2 * torch.pi / (2 * L - 1)


def quad_weight_dh_theta_only(theta: float, L: int) -> float:
    r"""Compute DH quadrature weight for :math:`\theta` integration (only), for given
    :math:`\theta`. Torch implementation of :func:`s2fft.quadrature.quad_weights_dh_theta_only`.

    Args:
        theta (float): :math:`\theta` angle for which to compute weight.

        L (int): Harmonic band-limit.

    Returns:
        float: Weight computed for each :math:`\theta`.
    """
    w = 0.0
    for k in range(0, L):
        w += torch.sin((2 * k + 1) * torch.from_numpy(theta)) / (2 * k + 1)

    w *= 2 / L * torch.sin(torch.from_numpy(theta))

    return w


def quad_weights_mw(L: int) -> torch.tensor:
    r"""Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.
    Torch implementation of :func:`s2fft.quadrature.quad_weights_mw`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    return quad_weights_mw_theta_only(L) * 2 * torch.pi / (2 * L - 1)


def quad_weights_mwss(L: int) -> torch.tensor:
    r"""Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.
    JAX implementation of :func:`s2fft.quadrature.quad_weights_mwss`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Weights computed for each :math:`\theta` (weights are identical
        as :math:`\phi` varies for given :math:`\theta`).
    """
    return quad_weights_mwss_theta_only(L) * 2 * torch.pi / (2 * L)


def quad_weights_mwss_theta_only(L: int) -> torch.tensor:
    r"""Compute MWSS quadrature weights for :math:`\theta` integration (only).
    Torch implementation of :func:`s2fft.quadrature.quad_weights_mwss_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Weights computed for each :math:`\theta`.
    """
    w = torch.zeros(2 * L, dtype=torch.complex128)

    for i in range(-(L - 1) + 1, L + 1):
        w[i + L - 1] = mw_weights(i - 1)

    wr = torch.real(torch.fft.fft(torch.fft.ifftshift(w), norm="backward")) / (2 * L)
    q = wr[: L + 1]
    q[1:L] += torch.flip(wr, dims=[0])[: L - 1]

    return q


def quad_weights_mw_theta_only(L: int) -> torch.tensor:
    r"""Compute MW quadrature weights for :math:`\theta` integration (only).
    Torch implementation of :func:`s2fft.quadrature.quad_weights_mw_theta_only`.

    Args:
        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Weights computed for each :math:`\theta`.
    """
    w = torch.zeros(2 * L - 1, dtype=torch.complex128)
    for i in range(-(L - 1), L):
        w[i + L - 1] = mw_weights(i)

    w *= torch.exp(-1j * torch.arange(-(L - 1), L) * torch.pi / (2 * L - 1))
    wr = torch.real(torch.fft.fft(torch.fft.ifftshift(w), norm="backward")) / (
        2 * L - 1
    )
    q = wr[:L]
    q[: L - 1] += torch.flip(wr, dims=[0])[: L - 1]

    return q


def mw_weights(m: int) -> float:
    r"""Compute MW weights given as a function of index m.

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
        return 1j * torch.pi / 2

    elif m == -1:
        return -1j * torch.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0
