import torch

from s2fft.sampling import s2_samples as samples


def mw_to_mwss(f_mw: torch.tensor, L: int, spin: int = 0) -> torch.tensor:
    r"""
    Convert signal on the sphere from MW sampling to MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.  Second, zero padding in harmonic space corresponding to
    :math:`\phi` is performed.

    Torch implementation of :func:`~s2fft.resampling.mw_to_mwss`.

    Args:
        f_mw (torch.tensor): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Signal on the sphere sampled with MWSS sampling.

    """
    if f_mw.ndim == 2:
        return torch.squeeze(
            mw_to_mwss_phi(mw_to_mwss_theta(torch.unsqueeze(f_mw, 0), L, spin), L)
        )
    else:
        return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)


def mw_to_mwss_theta(f_mw: torch.tensor, L: int, spin: int = 0) -> torch.tensor:
    r"""
    Convert :math:`\theta` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.

    Torch implementation of :func:`~s2fft.resampling.mw_to_mwss_theta`.


    Args:
        f_mw (torch.tensor): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Raises:
        ValueError: Input spherical signal must have shape matching MW sampling.

    Returns:
        torch.tensor: Signal on the sphere with MWSS sampling in :math:`\theta` and MW
        sampling in :math:`\phi`.

    """
    f_mw_ext = periodic_extension(f_mw, L, spin=spin, sampling="mw")
    fmp_mwss_ext = torch.zeros(
        (f_mw_ext.shape[0], 2 * L, 2 * L - 1), dtype=torch.complex128
    )

    fmp_mwss_ext[:, 1:, :] = torch.fft.fftshift(
        torch.fft.fft(f_mw_ext, axis=-2, norm="forward"), dim=[-2]
    )

    fmp_mwss_ext[:, 1:, :] = torch.einsum(
        "...blp,l->...blp",
        fmp_mwss_ext[:, 1:, :],
        torch.exp(
            -1j
            * torch.arange(-(L - 1), L, dtype=torch.float64)
            * torch.pi
            / (2 * L - 1)
        ),
    )

    f_mwss_ext = torch.conj(
        torch.fft.fft(
            torch.fft.ifftshift(torch.conj(fmp_mwss_ext), dim=[-2]),
            axis=-2,
            norm="backward",
        )
    )

    return unextend(f_mwss_ext, L, sampling="mwss")


def mw_to_mwss_phi(f_mw: torch.tensor, L: int) -> torch.tensor:
    r"""
    Convert :math:`\phi` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by zero padding in harmonic space.

    Torch implementation of :func:`~s2fft.resampling.mw_to_mwss_phi`.


    Note:
        Can work with arbitrary number of :math:`\theta` samples.  Hence, to convert
        both :math:`(\theta,\phi)` sampling to MWSS, can use :func:`~mw_to_mwss_theta`
        to first convert :math:`\theta` sampling before using this function to convert
        the :math:`\phi` sampling.

    Args:
        f_mw (torch.tensor): Signal on the sphere sampled with MW sampling in
            :math:`\phi` and arbitrary number of samples in

        L (int): Harmonic band-limit.

    Raises:
        ValueError: Input spherical signal must have number of samples in :math:`\phi`
            matching MW sampling.

    Returns:
        torch.tensor: Signal on the sphere with MWSS sampling in :math:`\phi` and
        sampling in :math:`\theta` of the input signal.

    """
    f_mwss = torch.zeros((f_mw.shape[0], L + 1, 2 * L), dtype=torch.complex128)
    f_mwss[:, :, 1:] = torch.fft.fftshift(
        torch.fft.fft(f_mw, axis=-1, norm="forward"), dim=[-1]
    )

    return torch.conj(
        torch.fft.fft(
            torch.fft.ifftshift(torch.conj(f_mwss), dim=[-1]),
            axis=-1,
            norm="backward",
        )
    )


def periodic_extension(
    f: torch.tensor, L: int, spin: int = 0, sampling: str = "mw"
) -> torch.tensor:
    r"""
    Perform period extension of MW/MWSS signal on the sphere in harmonic
    domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
    Torch implementation of :func:`~s2fft.resampling.periodic_extension`.

    Args:
        f (torch.tensor): Signal on the sphere sampled with MW/MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

    Returns:
        torch.tensor: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in same scheme (MW/MWSS) as input.

    """
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = samples.ntheta_extension(L, sampling)
    m_offset = 1 if sampling == "mwss" else 0

    f_ext = torch.zeros((f.shape[0], ntheta_ext, nphi), dtype=torch.complex128)
    f_ext[:, 0:ntheta, 0:nphi] = f[:, 0:ntheta, 0:nphi]
    f_ext = torch.fft.fftshift(torch.fft.fft(f_ext, dim=-1, norm="backward"), dim=[-1])

    f_ext[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ] = torch.flip(
        f_ext[
            :,
            m_offset : L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ],
        dims=[-2],
    )
    f_ext[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ] *= (-1) ** (torch.arange(-(L - 1), L))
    if hasattr(spin, "size"):
        f_ext[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ] = torch.einsum(
            "nlm,n->nlm",
            f_ext[
                :,
                L + m_offset : 2 * L - 1 + m_offset,
                m_offset : 2 * L - 1 + m_offset,
            ],
            (-1) ** spin,
        )
    else:
        f_ext[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ] *= (-1) ** spin

    return (
        torch.conj(
            torch.fft.fft(
                torch.fft.ifftshift(torch.conj(f_ext), dim=[-1]),
                axis=-1,
                norm="backward",
            )
        )
        / nphi
    )


def unextend(f_ext: torch.tensor, L: int, sampling: str = "mw") -> torch.tensor:
    r"""
    Unextend MW/MWSS sampled signal from :math:`\theta` domain
    :math:`[0,2\pi]` to :math:`[0,\pi]`.

    Args:
        f_ext (torch.tensor): Signal on the sphere sampled on extended :math:`\theta`
            domain :math:`[0,2\pi]`.

        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

        ValueError: Period extension must have correct shape.

    Returns:
        torch.tensor: Signal on the sphere sampled on :math:`\theta` domain
        :math:`[0,\pi]`.

    """
    if sampling.lower() == "mw":
        return f_ext[:, 0:L, :]

    elif sampling.lower() == "mwss":
        return f_ext[:, 0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )


def upsample_by_two_mwss(f: torch.tensor, L: int, spin: int = 0) -> torch.tensor:
    r"""
    Upsample MWSS sampled signal on the sphere defined on domain :math:`[0,\pi]`
    by a factor of two.

    Upsampling is performed by a periodic extension in :math:`\theta` to
    :math:`[0,2\pi]`, followed by zero-padding in harmonic space, followed by
    unextending :math:`\theta` domain back to :math:`[0,\pi]`.

    Torch implementation of :func:`~s2fft.resampling.upsample_by_two_mwss`.

    Args:
        f (torch.tensor): Signal on the sphere sampled with MWSS sampling scheme, sampled
            at resolution L.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Signal on the sphere sampled with MWSS sampling scheme, sampling at
        resolution 2*L.

    """
    if f.ndim == 2:
        f = torch.unsqueeze(f, 0)
    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext = upsample_by_two_mwss_ext(f_ext, L)
    f_ext = unextend(f_ext, 2 * L, sampling="mwss")
    return torch.squeeze(f_ext)


def upsample_by_two_mwss_ext(f_ext: torch.tensor, L: int) -> torch.tensor:
    r"""
    Upsample an extended MWSS sampled signal on the sphere defined on domain
    :math:`[0,2\pi]` by a factor of two.

    Upsampling is performed by zero-padding in harmonic space. Torch implementation of
    :func:`~s2fft.resampling.upsample_by_two_mwss_ext`.

    Args:
        f_ext (torch.tensor): Signal on the sphere sampled on extended MWSS sampling
            scheme on domain :math:`[0,2\pi]`, sampled at resolution L.

        L (int): Harmonic band-limit.

    Returns:
        torch.tensor: Signal on the sphere sampled on extended MWSS sampling scheme on
        domain :math:`[0,2\pi]`, sampling at resolution 2*L.

    """
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = torch.fft.fftshift(torch.fft.fft(f_ext, axis=-2, norm="forward"), dim=[-2])

    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = torch.zeros(
        (f_ext.shape[0], ntheta_ext_up, nphi), dtype=torch.complex128
    )
    f_ext_up[:, L : ntheta_ext + L, :nphi] = f_ext[:, 0:ntheta_ext, :nphi]
    return torch.conj(
        torch.fft.fft(
            torch.fft.ifftshift(torch.conj(f_ext_up), dim=[-2]),
            axis=-2,
            norm="backward",
        )
    )


def periodic_extension_spatial_mwss(
    f: torch.tensor, L: int, spin: int = 0
) -> torch.tensor:
    r"""
    Perform period extension of MWSS signal on the sphere in spatial domain,
    extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.

    For the MWSS sampling scheme, it is possible to do the period extension in
    :math:`\theta` in the spatial domain.  This is not possible for the MW sampling
    scheme.

    Torch implementation of :func:`~s2fft.resampling.periodic_extension_spatial_mwss`.

    Args:
        f (torch.tensor): Signal on the sphere sampled with MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        torch.tensor: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in MWSS sampling scheme.

    """
    ntheta = L + 1
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = torch.zeros((f.shape[0], ntheta_ext, nphi), dtype=torch.complex128)
    f_ext[:, 0:ntheta, 0:nphi] = f[:, 0:ntheta, 0:nphi]
    if hasattr(spin, "size"):
        f_ext[:, ntheta:, 0 : 2 * L] = torch.einsum(
            "btp,b->btp",
            torch.fft.fftshift(
                torch.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], dims=[-2]), dim=[-1]
            ),
            (-1) ** spin,
        )
    else:
        f_ext[:, ntheta:, 0 : 2 * L] = (-1) ** spin * torch.fft.fftshift(
            torch.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], dims=[-2]), dim=[-1]
        )
    return f_ext
