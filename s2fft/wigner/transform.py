import numpy as np
import numpy.fft as fft
import s2fft.transform as s2f
import s2fft.wigner.samples as samples

# TODO: Extend to other sampling schemes, only "mw" supported.
# TODO: Extend to support efficient storage methods, only "padded" supported.
# TODO: Extend to support symmetry accelerations, no symmetries exploited currently.
def inverse_wigner_transform(
    flmn: np.ndarray, L: int, N: int = 1, sampling: str = "mw"
) -> np.ndarray:
    r"""Function to compute the inverse Wigner transform, i.e. inverse Fourier transform on :math:`SO(3)`.

    Args:
        flmn (np.ndarray): Array of Wigner coefficients with shape :math:`[L, 2L-1, N]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations
            (i.e. directionality). Defaults to 1.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        (np.ndarray): Pixel-space samples of the rotation group with shape :math:`[n_{\beta}, n_{\alpha}, n_{\gamma}]`.
    """
    assert flmn.shape == samples.flmn_shape(L, N)

    if sampling not in ["mw", "mwss", "dh"]:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    fabn = np.zeros(samples.f_shape(L, N, sampling), dtype=np.complex128)

    for el in range(L):
        flmn[el, ...] *= np.sqrt((2 * el + 1) / (16 * np.pi**3))

    for n in range(-N + 1, N):
        fabn[..., N - 1 + n] = (-1) ** n * s2f.inverse_sov_fft(
            flmn[..., N - 1 + n], L, spin=-n, sampling=sampling
        )

    return fft.ifft(fft.ifftshift(fabn, axes=2), axis=2, norm="forward")


def forward_wigner_transform(
    f: np.ndarray, L: int, N: int = 1, sampling: str = "mw"
) -> np.ndarray:
    r"""Function to compute the forward Wigner transform, i.e. Fourier transform on :math:`SO(3)`.

    Args:
        f (np.ndarray): Array of samples on :math:`SO(3)` with shape :math:`[n_{\alpha}, n_{\beta}, n_{\gamma}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations
            (i.e. directionality). Defaults to 1.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Sampling scheme not currently supported.

    Returns:
        (np.ndarray): Wigner coefficients with shape :math:`[L, 2L-1, 2N-1]`.
    """
    assert f.shape == samples.f_shape(L, N, sampling)

    if sampling not in ["mw", "mwss", "dh"]:
        raise ValueError(f"Sampling scheme sampling={sampling} not supported")

    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)

    fabn = fft.fftshift(fft.fft(f, axis=2, norm="backward"), axes=2)
    fabn *= 2 * np.pi / (2 * N - 1)

    for n in range(-N + 1, N):
        flmn[..., N - 1 + n] = (-1) ** n * s2f.forward_sov_fft(
            fabn[..., N - 1 + n], L, spin=-n, sampling=sampling
        )

    for el in range(L):
        flmn[el, ...] *= np.sqrt(4 * np.pi / (2 * el + 1))

    return flmn
