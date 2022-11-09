from random import sample
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

import jax
# from jax import jit, device_put
import jax.numpy as jnp

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

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= spin < L

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

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= spin < L

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

    assert flm.shape == samples.flm_shape(L)
    assert 0 <= spin < L

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


def forward_direct( # Direct method (?)
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

    # size checks
    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    # only takes one sampling scheme (dh)
    if sampling.lower() != "dh":
        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    # initialise array with spherical harmonics
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128) # 2D array of shape (L, 2 * L - 1)

    # build arrays of thetas and phis according to sampling
    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)
    # get weights for this L and sampling scheme
    weights = quadrature.quad_weights(L, sampling)

    # loop thru thetas
    for t, theta in enumerate(thetas):

        # loop thru Harmonic deg (what is el? Harmonic degree of Wigner-d matrix, ranges from 0 to L, harmonic band-limit)
        for el in range(0, L):

            # if el
            if el >= np.abs(spin):

                # compute slice of Wigned d-matrix for this theta and Harmonic deg
                dl = wigner.turok.compute_slice(theta, el, L, -spin) # returns 1D array of size [2L-1]?

                # compute 'elfactor' for this el
                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                # for a narrow range of el?
                for m in range(-el, el + 1): # -el, ..0,... el, el+1

                    for p, phi in enumerate(phis_equiang):

                        flm[el, m + L - 1] += (
                            weights[t] # weight from sampling scheme, for this theta
                            * (-1) ** spin # spin
                            * elfactor # Harmonic degree factor
                            * np.exp(-1j * m * phi) # complex
                            * dl[m + L - 1] # take relevant part of the slice
                            * f[t, p] # signal on the sphere at theta elem t and phi element p
                        ) # JAX at operator? vmap?

    return flm


def forward_sov( # with separation of variables (no FFT)
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

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    if sampling.lower() != "dh":
        raise ValueError(
            f"Sampling scheme sampling={sampling} not implement (only DH supported at present)"
        )

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)  # 2D array of shape (L, 2 * L - 1)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ntheta = samples.ntheta(L, sampling) # is this same as len(thetas)?

    # compute ftm
    ftm = np.zeros((ntheta, 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        for m in range(-(L - 1), L):

            for p, phi in enumerate(phis_equiang):

                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[t, p]

    # compute flm (using ftm matrix)
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


def forward_sov_fft( # with separation of variables and FFT --how is this with fft and the other without?
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

    assert f.shape == samples.f_shape(L, sampling)
    assert 0 <= spin < L

    # sample f and theta as required
    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)

    # initialise flm, ftm
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

    ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1) #FFT to input signal?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi)) # make agnostic (**0.5)

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            )

    flm *= (-1) ** spin

    return flm

# jit?
def forward_sov_fft_vectorized_jax( # vectorised
    f: np.ndarray, L: int, 
    spin: int = 0, sampling: str = "mw", device: str = jax.devices()[0]
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

    # transform f to DeviceArray and commit to device---all the rest DeviceArrays and committed too? check
    f = jax.device_put(f, device)

    # transform to DeviceArrays and put all arrays in default device?
    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling)
    thetas = jax.device_put(thetas) #float64

    # initialise flm and ftm matrices and commit to device
    flm = jax.device_put(jnp.zeros(samples.flm_shape(L), dtype=np.complex128), device) # initialise array and put in device
    ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1) #FFT to input signal? Use JAX implementation!--should be in device already?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = jax.device_put(quadrature.quad_weights_transform(L, sampling, spin=0), device) #put in device? dtype?
    m_offset = 1 if sampling == "mwss" else 0

    #----------------
    # el_array = jnp.array(range(spin, L))
    # for t, theta in enumerate(thetas):

    #     for el in range(spin, L):

    #         dl = wigner.turok.compute_slice(theta, el, L, -spin) #--vmap for all thetas? and all el

    #         flm.at[el, :].add(jnp.prod(weights[t],
    #                                    ((2 * el_array + 1) / (4 * jnp.pi))**0.5, 
    #                                        dl, 
    #                                        ftm[t, m_offset : 2 * L - 1 + m_offset])) #broadcast?

    #         # elfactor = np.sqrt((2 * el + 1) / (4 * np.pi)) # make agnostic (**0.5)---vmap for el?

    #         # flm[el, :] += (np.multiply(weights[t],
    #         #                             np.sqrt((2 * el_array + 1) / (4 * np.pi)), 
    #         #                             dl, 
    #         #                             ftm[t, m_offset : 2 * L - 1 + m_offset]) #broadcast?
    #         # ) # at operator! or better: sum at the end

    # flm = flm*((-1) ** spin)
    #----------------

    #----------------
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi)) # make agnostic (**0.5)

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            ) # at operator! or better: sum at the end

    flm *= (-1) ** spin
    #----------------

    return flm
