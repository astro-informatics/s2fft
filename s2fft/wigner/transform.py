import numpy as np
import numpy.fft as fft
from s2fft.spherical import transform as s2transform
from s2fft.wigner import samples


def inverse(flmn: np.ndarray, L: int, N: int, sampling: str = "mw") -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\beta, \alpha,
        \gamma]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[L, 2L-1, N]`.

        L (int): Harmonic bandlimit.

        N (int): Directional band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\beta},
        n_{\alpha}, n_{\gamma}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.
    """
    assert flmn.shape == samples.flmn_shape(L, N)

    if sampling not in ["mw", "mwss", "dh"]:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    fban = np.zeros(samples.f_shape(L, N, sampling), dtype=np.complex128)

    flmn_scaled = np.einsum(
        "ijk,i->ijk", flmn, np.sqrt((2 * np.arange(L) + 1) / (16 * np.pi**3))
    )

    for n in range(-N + 1, N):
        fban[..., N - 1 + n] = (-1) ** n * s2transform.inverse(
            flmn_scaled[..., N - 1 + n], L, spin=-n, sampling=sampling
        )

    f = fft.ifft(fft.ifftshift(fban, axes=2), axis=2, norm="forward")

    return f


def forward(f: np.ndarray, L: int, N: int, sampling: str = "mw") -> np.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\beta, \alpha,
        \gamma]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\beta}, n_{\alpha}, n_{\gamma}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic bandlimit.

        N (int): Directional band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[L, 2L-1, 2N-1]`.
    """
    assert f.shape == samples.f_shape(L, N, sampling)

    if sampling not in ["mw", "mwss", "dh"]:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)

    fabn = (
        2
        * np.pi
        / (2 * N - 1)
        * fft.fftshift(fft.fft(f, axis=2, norm="backward"), axes=2)
    )

    for n in range(-N + 1, N):
        flmn[..., N - 1 + n] = (-1) ** n * s2transform.forward(
            fabn[..., N - 1 + n], L, spin=-n, sampling=sampling
        )

    flmn = np.einsum("ijk,i->ijk", flmn, np.sqrt(4 * np.pi / (2 * np.arange(L) + 1)))

    return flmn
