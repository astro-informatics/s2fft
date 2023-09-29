import numpy as np
from s2fft.sampling import so3_samples as samples
from s2fft.base_transforms import spherical


def inverse(
    flmn: np.ndarray,
    L: int,
    N: int,
    L_lower: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    nside: int = None,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
        \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic bandlimit.

        N (int): Directional band-limit.

        L_lower (int, optional): Harmonic lower bound. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\gamma},
        n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.
    """
    assert flmn.shape == samples.flmn_shape(L, N)

    fban = np.zeros(samples.f_shape(L, N, sampling, nside), dtype=np.complex128)

    flmn_scaled = np.einsum(
        "...nlm,...l->...nlm",
        flmn,
        np.sqrt((2 * np.arange(L) + 1) / (16 * np.pi**3)),
    )

    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        fban[N - 1 + n] = (-1) ** n * spherical.inverse(
            flmn_scaled[N - 1 + n],
            L,
            spin=-n,
            sampling=sampling,
            reality=reality,
            L_lower=L_lower,
            nside=nside,
        )

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        f = np.fft.irfft(fban[N - 1 :], 2 * N - 1, axis=ax, norm="forward")
    else:
        f = np.fft.ifft(np.fft.ifftshift(fban, axes=ax), axis=ax, norm="forward")

    return f


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    L_lower: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    nside: int = None,
) -> np.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
        \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic bandlimit.

        N (int): Directional band-limit.

        L_lower (int, optional): Harmonic lower bound. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.
    """
    assert f.shape == samples.f_shape(L, N, sampling, nside)

    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        fban = np.fft.rfft(np.real(f), axis=ax, norm="backward")
    else:
        fban = np.fft.fftshift(np.fft.fft(f, axis=ax, norm="backward"), axes=ax)

    fban *= 2 * np.pi / (2 * N - 1)

    if reality:
        sgn = (-1) ** abs(np.arange(-L + 1, L))

    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        flmn[N - 1 + n] = (-1) ** n * spherical.forward(
            fban[n - n_start_ind],
            L,
            spin=-n,
            sampling=sampling,
            reality=reality,
            L_lower=L_lower,
            nside=nside,
        )
        if reality and n != 0:
            flmn[N - 1 - n] = np.conj(
                np.flip(flmn[N - 1 + n] * sgn * (-1) ** n, axis=-1)
            )

    flmn = np.einsum(
        "...nlm,...l->...nlm", flmn, np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))
    )

    return flmn
