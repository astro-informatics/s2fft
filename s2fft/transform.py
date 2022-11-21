import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial

import numpy.fft as fft
import jax.numpy.fft as jfft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner


def inverse_direct(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by direct method.

    Warning:
        This implmentation is very slow and intended for testing purposes only.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
    """

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= np.abs(spin) < L

    if sampling.lower() != "healpix":

        phis_ring = samples.phis_equiang(L, sampling)

    thetas = samples.thetas(L, sampling, nside)
    f = np.zeros(samples.f_shape(L, sampling, nside), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    if sampling.lower() == "healpix":
                        phis_ring = samples.phis_ring(t, nside)

                    for p, phi in enumerate(phis_ring):

                        if sampling.lower() != "healpix":
                            entry = (t, p)

                        else:
                            entry = samples.hp_ang2pix(nside, theta, phi)

                        f[entry] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1]
                            * flm[el, m + L - 1]
                        )

    return f


def inverse_sov(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
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
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
    """

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= np.abs(spin) < L

    ntheta = samples.ntheta(L, sampling, nside=nside)
    thetas = samples.thetas(L, sampling, nside=nside)

    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    f = np.zeros(samples.f_shape(L, sampling, nside), dtype=np.complex128)

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

        if sampling.lower() == "healpix":
            phis_ring = samples.phis_ring(t, nside)

        for p, phi in enumerate(phis_ring):

            for m in range(-(L - 1), L):

                if sampling.lower() != "healpix":
                    entry = (t, p)

                else:
                    entry = samples.hp_ang2pix(nside, theta, phi)

                f[entry] += ftm[t, m + L - 1] * np.exp(1j * m * phi)

    return f


def inverse_sov_fft(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by separate of variables method
    with FFTs.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
    """

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= np.abs(spin) < L
    if sampling.lower() == "healpix":
        assert L >= 2 * nside

    thetas = samples.thetas(L, sampling, nside)

    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):
        if sampling.lower() == "healpix":
            phi_ring_offset = samples.p2phi_ring(t, 0, nside)

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    phase_shift = (
                        np.exp(1j * m * phi_ring_offset)
                        if sampling.lower() == "healpix"
                        else 1
                    )

                    ftm[t, m + L - 1 + m_offset] += (
                        (-1) ** spin * elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                    ) * phase_shift

    if sampling.lower() == "healpix":
        f = resampling.healpix_ifft(ftm, L, nside)
    else:
        f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def inverse_sov_fft_vectorized(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute inverse spherical harmonic transform by separate of variables method
    with FFTs (vectorized implementaiton).

    Args:
        flm (np.ndarray): Spherical harmonic coefficients

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Signal on the sphere.
    """

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= np.abs(spin) < L

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
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
) -> np.ndarray:
    """Compute forward spherical harmonic transform by direct method.

    Warning:
        This implmentation is very slow and intended for testing purposes only.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    assert f.shape == samples.f_shape(L, sampling, nside)
    assert 0 <= np.abs(spin) < L

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)

    else:
        thetas = samples.thetas(L, sampling, nside)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)

    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    for t, theta in enumerate(thetas):

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    if sampling.lower() == "healpix":
                        phis_ring = samples.phis_ring(t, nside)

                    for p, phi in enumerate(phis_ring):

                        if sampling.lower() != "healpix":
                            entry = (t, p)
                        else:
                            entry = samples.hp_ang2pix(nside, theta, phi)

                        flm[el, m + L - 1] += (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1]
                            * f[entry]
                        )

    return flm


def forward_sov(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
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
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.


    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    assert f.shape == samples.f_shape(L, sampling, nside)
    assert 0 <= np.abs(spin) < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)

    else:
        thetas = samples.thetas(L, sampling, nside)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    ftm = np.zeros((len(thetas), 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            if sampling.lower() == "healpix":
                phis_ring = samples.phis_ring(t, nside)

            for p, phi in enumerate(phis_ring):

                if sampling.lower() != "healpix":
                    entry = (t, p)
                else:
                    entry = samples.hp_ang2pix(nside, theta, phi)

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[entry]

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)

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
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    assert f.shape == samples.f_shape(L, sampling, nside)
    assert 0 <= np.abs(spin) < L
    if sampling.lower() == "healpix":
        assert L >= 2 * nside

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling, nside)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if sampling.lower() == "healpix":
        ftm = resampling.healpix_fft(f, L, nside)
    else:
        ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):
        if sampling.lower() == "healpix":
            phi_ring_offset = samples.p2phi_ring(t, 0, nside)

        for el in range(0, L):

            if el >= np.abs(spin):

                dl = wigner.turok.compute_slice(theta, el, L, -spin)

                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                for m in range(-el, el + 1):

                    phase_shift = (
                        np.exp(-1j * m * phi_ring_offset)
                        if sampling.lower() == "healpix"
                        else 1
                    )

                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                    ) * phase_shift

    return flm


def forward_sov_fft_vectorized(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs (vectorized implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= np.abs(spin) < L

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

        for el in range(spin, L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            )

    flm *= (-1) ** spin

    return flm
