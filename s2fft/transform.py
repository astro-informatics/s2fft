from random import sample
import numpy as np
import numpy.fft as fft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner


def inverse_direct(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by direct method.

    Warning:
        This implmentation is very slow and intended for testing purposes only.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Signal on the sphere.
    """

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(phis_equiang):

                        f[t, p] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1]
                            * flm[el, m + L - 1]
                        )

    return f


def inverse_sov(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by separate of variables method
    (without FFTs).

    Warning:
        This implmentation is intended for testing purposes only.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Signal on the sphere.
    """

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    ftm[t, m + L - 1] += (
                        (-1) ** spin * elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                    )

    for t, theta in enumerate(thetas):

        for p, phi in enumerate(phis_equiang):

            for m in range(-(L - 1), L):

                f[t, p] += ftm[t, m + L - 1] * np.exp(1j * m * phi)

    return f


def inverse_sov_fft(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by separate of variables method
    with FFTs.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Signal on the sphere.
    """

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ftm = np.zeros((ntheta, nphi), dtype=np.complex128)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    ftm[t, m + L - 1 + m_offset] += (
                        (-1) ** spin * elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                    )

    f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def inverse_sov_fft_vect(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:

    # TODO: Check flm shape consistent with L

    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    f = np.zeros((ntheta, nphi), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ftm = np.zeros((ntheta, nphi), dtype=np.complex128)
    m_offset = 1 if sampling == "mwss" else 0
    for el in range(spin, L):

        for t, theta in enumerate(thetas):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            ftm[t, m_offset : 2 * L - 1 + m_offset] += elfactor * np.multiply(
                dl, flm[el, :]
            )

    ftm *= (-1) ** (spin)
    f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def forward_direct(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute forward spherical harmonic transform by direct method.

    Warning:
        This implmentation is very slow and intended for testing purposes only.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Only DH sampling supported at present.

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    # TODO: Check f shape consistent with L

    if sampling.lower() != "dh":

        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    weights = quadrature.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    for p, phi in enumerate(phis_equiang):

                        flm[el, m + L - 1] += (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1]
                            * f[t, p]
                        )

    return flm


def forward_sov(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    (without FFTs).

    Warning:
        This implmentation is intended for testing purposes only.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Raises:
        ValueError: Only DH sampling supported at present.

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    # TODO: Check f shape consistent with L

    if sampling.lower() != "dh":

        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ntheta = samples.ntheta(L, sampling)
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[t, p]

    weights = quadrature.quad_weights(L, sampling)
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1]
                    )

    return flm


def forward_sov_fft(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    # TODO: Check f shape consistent with L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                    )

    return flm
