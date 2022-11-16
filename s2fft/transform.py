from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
import numpy.fft as fft
from jax import jit

import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.samples as samples
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
    assert 0 <= spin < L

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
    assert 0 <= spin < L

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
    assert 0 <= spin < L
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
    assert 0 <= spin < L

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
    assert 0 <= spin < L

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

        # loop thru Harmonic deg (what is el? Harmonic degree of Wigner-d matrix, ranges from 0 to L, harmonic band-limit)
        for el in range(0, L):

            # if el
            if el >= np.abs(spin):

                # compute slice of Wigned d-matrix for this theta and Harmonic deg
                dl = wigner.turok.compute_slice(
                    theta, el, L, -spin
                )  # returns 1D array of size [2L-1]?

                # compute 'elfactor' for this el
                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                # for a narrow range of el?
                for m in range(-el, el + 1):  # -el, ..0,... el, el+1

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
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)

    else:
        thetas = samples.thetas(L, sampling, nside)

    flm = np.zeros(
        samples.flm_shape(L), dtype=np.complex128
    )  # 2D array of shape (L, 2 * L - 1)

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
    assert 0 <= spin < L
    if sampling.lower() == "healpix":
        assert L >= 2 * nside

    # sample f and theta as required
    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling, nside)

    # initialise flm, ftm
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
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    ftm = fft.fftshift(
        fft.fft(f, axis=1, norm="backward"), axes=1
    ) 

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


def forward_sov_fft_vectorized_jax_turok(
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
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    ftm = fft.fftshift(
        fft.fft(f, axis=1, norm="backward"), axes=1
    ) 

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = jnp.flip(wigner.turok_jax.compute_slice(theta, el, L, -spin)) #=---------------------

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))  

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            )

    flm *= (-1) ** spin

    return flm

def forward_sov_fft_vectorized_00(
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
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    flm = jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128)

    ftm = jnp.fft.fftshift(
        jnp.fft.fft(f, axis=1, norm="backward"), axes=1
    ) 

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = wigner.turok_jax.compute_slice(theta, el, L, -spin)

            elfactor = jnp.sqrt((2 * el + 1) / (4 * jnp.pi))  

            flm.at[el, :].add(
                weights[t]
                * elfactor
                * jnp.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            )

    flm.at[:,:].multiply((-1) ** spin)

    return flm

def forward_sov_fft_vectorized_jax_0(
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
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    f = jax.device_put(f)
    thetas = jax.device_put(thetas)

    flm = jnp.zeros(samples.flm_shape(L), dtype=np.complex128)

    ftm = jnp.fft.fftshift(
        jnp.fft.fft(f, axis=1, norm="backward"), axes=1
    )  # FFT to input signal?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = jnp.array(quadrature.quad_weights_transform(L, sampling, spin=0))
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = jnp.array(wigner.turok_jax.compute_slice(theta, el, L, -spin))

            elfactor = jnp.sqrt((2 * el + 1) / (4 * jnp.pi))  # make agnostic (**0.5)

            flm.at[el, :].add(
                jnp.broadcast_to(weights[t],dl.shape)
                * jnp.broadcast_to(elfactor,dl.shape)
                * jnp.multiply(dl, jax.lax.slice_in_dim(ftm.at[t,:].get(), m_offset, 2 * L - 1 + m_offset, axis=-1))
            )

    flm.at[:,:].multiply((-1) ** spin)

    return flm

# jit?
def forward_sov_fft_vectorized_jax_1(  # vectorised
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    device: str = jax.devices()[0],
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs (vectorized implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

        device (str, optional): JAX device to place data in. Default is jax.devices()[0],
            which is a GPU if present, otherwise CPU

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    # transform to DeviceArray and commit to device
    f = jax.device_put(f, device)
    thetas = jax.device_put(thetas)

    # ftm array
    ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1)  # same device as f right?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = jax.device_put(
        quadrature.quad_weights_transform(L, sampling, spin=0), device
    )
    m_offset = 1 if sampling == "mwss" else 0

    # el array
    # el_array = jnp.array(
    #     range(spin, L), dtype=np.int64
    # )  # needs to be int for wigner.turok_jax.compute_slice
    # el_array_fl = jnp.array(range(spin, L), dtype=np.float64)  # to prevent el_factor_3D.weak_type=True

    # Compute dl_vmapped fn
    dl_vmapped = jax.vmap(
        jax.vmap(
            wigner.turok_jax.compute_slice,
            in_axes=(0, None, None, None),
            out_axes=-1,
        ),
        in_axes=(None, 0, None, None),
        out_axes=0,
    )

    # Compute flm
    flm = (
        (
            jnp.expand_dims(weights, axis=(0, 1))  # weights[None, None, :] --agnostic but seems slower?
            * jnp.expand_dims(
                jnp.sqrt((2 * jnp.array(range(spin, L), dtype=np.float64) + 1) / (4 * jnp.pi)),
                axis=(-1, -2),
            )
            * dl_vmapped(thetas, jnp.array(range(spin, L), dtype=np.int64), L, -spin)
            * jnp.expand_dims(
                jax.lax.slice_in_dim(ftm, m_offset, 2 * L - 1 + m_offset, axis=-1),
                axis=-1,
            ).T  # ftm[:, m_offset : 2 * L - 1 + m_offset, None].T
        )
        .sum(axis=-1)
        .at[:]
        .multiply((-1) ** spin)
    )

    # Pad with zeros
    flm = jnp.pad(flm, ((spin, 0), (0, 0)))

    return flm


# jit?
def forward_sov_fft_vectorized_jax_2(  # broadcasting more readable?
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    device: str = jax.devices()[0],
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs (vectorized implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

        device (str, optional): JAX device to place data in. Default is jax.devices()[0],
            which is a GPU if present, otherwise CPU

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    # transform to DeviceArray and commit to device
    f = jax.device_put(f, device)
    thetas = jax.device_put(thetas)

    # ftm array
    ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = jax.device_put(
        quadrature.quad_weights_transform(L, sampling, spin=0), device
    )
    m_offset = 1 if sampling == "mwss" else 0

    # el array
    el_array = jnp.array(
        range(spin, L), dtype=np.int64
    )  # needs to be int for wigner.turok_jax.compute_slice
    el_array_fl = jnp.array(
        el_array, dtype=np.float64
    )  # to prevent el_factor_3D.weak_type=True

    # Compute dl_vmapped fn
    dl_vmapped = jax.vmap(
        jax.vmap(
            wigner.turok_jax.compute_slice,
            in_axes=(0, None, None, None),
            out_axes=-1,
        ),
        in_axes=(None, 0, None, None),
        out_axes=0,
    )

    # Compute flm
    target_size_3D = (
        len(range(spin, L)),  # samples.flm_shape(L)[0]-spin?
        samples.flm_shape(L)[-1],
        len(thetas),
    )

    flm = (
        (
            jnp.broadcast_to(weights, target_size_3D)
            * (((2 * el_array_fl + 1) / (4 * jnp.pi)) ** 0.5)[:, None, None]
            * dl_vmapped(thetas, el_array, L, -spin)
            * jnp.broadcast_to(
                jax.lax.slice_in_dim(ftm, m_offset, 2 * L - 1 + m_offset, axis=-1).T,
                target_size_3D,
            )
        )
        .sum(axis=-1)
        .at[:]
        .multiply((-1) ** spin)
    )

    # Pad with zeros
    flm = jnp.pad(flm, ((spin, 0), (0, 0)))

    return flm


def forward_sov_fft_vectorized_jax_3(  # jnp.stack (worse for memory?)
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    device: str = jax.devices()[0],
) -> np.ndarray:
    """Compute forward spherical harmonic transform by separate of variables method
    with FFTs (vectorized implementation).

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}.  Defaults to "mw".

        device (str, optional): JAX device to place data in. Default is jax.devices()[0],
            which is a GPU if present, otherwise CPU

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    # transform to DeviceArray and commit to device
    f = jax.device_put(f, device)
    thetas = jax.device_put(thetas)

    # ftm array
    ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1)

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = jax.device_put(
        quadrature.quad_weights_transform(L, sampling, spin=0), device
    )
    m_offset = 1 if sampling == "mwss" else 0

    # el array
    el_array = jnp.array(
        range(spin, L), dtype=np.int64
    )  # needs to be int for wigner.turok_jax.compute_slice
    el_array_fl = jnp.array(
        el_array, dtype=np.float64
    )  # to prevent el_factor_3D.weak_type=True

    # Compute dl_vmapped fn
    dl_vmapped = jax.vmap(
        jax.vmap(
            wigner.turok_jax.compute_slice,
            in_axes=(0, None, None, None),
            out_axes=-1,
        ),
        in_axes=(None, 0, None, None),
        out_axes=0,
    )

    # Compute flm
    target_size_3D = (
        len(range(spin, L)),  # samples.flm_shape(L)[0]-spin?
        samples.flm_shape(L)[-1],
        len(thetas),
    )

    flm = (
        jnp.stack(
            (
                jnp.broadcast_to(weights, target_size_3D),
                jnp.broadcast_to(
                    (((2 * el_array_fl + 1) / (4 * jnp.pi)) ** 0.5)[:, None, None],
                    target_size_3D,
                ),
                dl_vmapped(thetas, el_array, L, -spin),
                jnp.broadcast_to(
                    jax.lax.slice_in_dim(
                        ftm, m_offset, 2 * L - 1 + m_offset, axis=-1
                    ).T,
                    target_size_3D,
                ),
            ),
            axis=-1,
        )
        .prod(axis=-1)
        .sum(axis=-1)
        .at[:]
        .multiply((-1) ** spin)
    )

    # Pad with zeros
    flm = jnp.pad(flm, ((spin, 0), (0, 0)))

    return flm
