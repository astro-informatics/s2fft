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


def forward_direct(  # Direct method (?)
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
    flm = np.zeros(
        samples.flm_shape(L), dtype=np.complex128
    )  # 2D array of shape (L, 2 * L - 1)

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
                dl = wigner.turok.compute_slice(
                    theta, el, L, -spin
                )  # returns 1D array of size [2L-1]?

                # compute 'elfactor' for this el
                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

                # for a narrow range of el?
                for m in range(-el, el + 1):  # -el, ..0,... el, el+1

                    for p, phi in enumerate(phis_equiang):

                        flm[el, m + L - 1] += (
                            weights[t]  # weight from sampling scheme, for this theta
                            * (-1) ** spin  # spin
                            * elfactor  # Harmonic degree factor
                            * np.exp(-1j * m * phi)  # complex
                            * dl[m + L - 1]  # take relevant part of the slice
                            * f[
                                t, p
                            ]  # signal on the sphere at theta elem t and phi element p
                        )  # JAX at operator? vmap?

    return flm


def forward_sov(  # with separation of variables (no FFT)
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

    flm = np.zeros(
        samples.flm_shape(L), dtype=np.complex128
    )  # 2D array of shape (L, 2 * L - 1)

    thetas = samples.thetas(L, sampling)
    phis_equiang = samples.phis_equiang(L, sampling)

    ntheta = samples.ntheta(L, sampling)  # is this same as len(thetas)?

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


def forward_sov_fft(  # with separation of variables and FFT --how is this with fft and the other without?
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

    ftm = fft.fftshift(
        fft.fft(f, axis=1, norm="backward"), axes=1
    )  # FFT to input signal?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))  # make agnostic (**0.5)

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
    )  # FFT to input signal?

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, spin=0)
    m_offset = 1 if sampling == "mwss" else 0
    for t, theta in enumerate(thetas):

        for el in range(spin, L):

            dl = wigner.turok_jax.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))  # make agnostic (**0.5)

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
            )

    flm *= (-1) ** spin

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
    ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1)

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
            jnp.expand_dims(weights, axis=(0, 1))  # weights[None, None, :]
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
