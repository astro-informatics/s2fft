import numpy as np
import numpy.fft as fft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner
import s2fft.healpix_ffts as hp

import jax
from jax import jit
# import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from functools import partial

def inverse(
    flm: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    """Compute inverse spherical harmonic transform.

    Uses a vectorised separation of variables method with FFT.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    return _inverse(flm, L, spin, sampling, nside=nside, method="sov_fft_vectorized")


def _inverse(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "sov_fft",
    nside: int = None,
) -> np.ndarray:
    """Compute inverse spherical harmonic transform using a specified method.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Harmonic transform algorithm. Supported algorithms include
            {"direct", "sov", "sov_fft", "sov_fft_vectorized"}. Defaults to "sov_fft".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    assert flm.shape == samples.flm_shape(L)
    assert 0 <= np.abs(spin) < L
    thetas = samples.thetas(L, sampling, nside)
    transform_methods = {
        "direct": _compute_inverse_direct,
        "sov": _compute_inverse_sov,
        "sov_fft": _compute_inverse_sov_fft,
        "sov_fft_vectorized": _compute_inverse_sov_fft_vectorized,
    }
    return transform_methods[method](flm, L, spin, sampling, thetas, nside=nside)


def forward(
    f: np.ndarray, L: int, spin: int = 0, sampling: str = "mw", nside: int = None
) -> np.ndarray:
    """Compute forward spherical harmonic transform.

    Uses a vectorised separation of variables method with FFT.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Spheircal harmonic coefficients.
    """
    return _forward(f, L, spin, sampling, nside=nside, method="sov_fft_vectorized")


def _forward(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "sov_fft",
    nside: int = None,
):
    """Compute forward spherical harmonic transform using a specified method.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Harmonic transform algorithm. Supported algorithms include
            {"direct", "sov", "sov_fft", "sov_fft_vectorized"}. Defaults to "sov_fft".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        np.ndarray: Signal on the sphere.
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

    # Don't need to include spin in weights (even for spin signals)
    # since accounted for already in periodic extension and upsampling.
    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)

    transform_methods = {
        "direct": _compute_forward_direct,
        "sov": _compute_forward_sov,
        "sov_fft": _compute_forward_sov_fft,
        "sov_fft_vectorized": _compute_forward_sov_fft_vectorized,
        "sov_fft_vectorized_jax_vmap": _compute_forward_sov_fft_vectorized_jax_vmap,
        "sov_fft_vectorized_jax_map": _compute_forward_sov_fft_vectorized_jax_map,
    }
    return transform_methods[method](f, L, spin, sampling, thetas, weights, nside=nside)


def _compute_inverse_direct(
    flm: np.ndarray, L: int, spin: int, sampling: str, thetas: np.ndarray, nside: int
):
    r"""Compute inverse spherical harmonic transform directly.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Signal on the sphere.
    """
    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

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


def _compute_inverse_sov(
    flm: np.ndarray, L: int, spin: int, sampling: str, thetas: np.ndarray, nside: int
):
    r"""Compute inverse spherical harmonic transform by separation of variables with a
        manual Fourier transform.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Signal on the sphere.
    """
    ftm = np.zeros((len(thetas), 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):
        for el in range(0, L):
            if el >= np.abs(spin):
                dl = wigner.turok.compute_slice(theta, el, L, -spin)
                elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))
                for m in range(-el, el + 1):
                    ftm[t, m + L - 1] += (
                        (-1) ** spin * elfactor * dl[m + L - 1] * flm[el, m + L - 1]
                    )

    f = np.zeros(samples.f_shape(L, sampling, nside), dtype=np.complex128)
    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)
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


def _compute_inverse_sov_fft(
    flm: np.ndarray, L: int, spin: int, sampling: str, thetas: np.ndarray, nside: int
):
    r"""Compute inverse spherical harmonic transform by separation of variables with a
        Fast Fourier transform.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Signal on the sphere.
    """

    if sampling.lower() == "healpix":
        assert L >= 2 * nside

    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):

        phi_ring_offset = (
            samples.p2phi_ring(t, 0, nside) if sampling.lower() == "healpix" else 0
        )

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
                        (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * flm[el, m + L - 1]
                        * phase_shift
                    )

    if sampling.lower() == "healpix":
        f = hp.healpix_ifft(ftm, L, nside)
    else:
        f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def _compute_inverse_sov_fft_vectorized(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    nside: int = None,
):
    r"""A vectorized function to compute inverse spherical harmonic transform by
        separation of variables with a manual Fourier transform.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Signal on the sphere.
    """
    ftm = np.zeros(samples.ftm_shape(L, sampling, nside), dtype=np.complex128)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):

        phase_shift = (
            samples.ring_phase_shift_hp(L, t, nside, False)
            if sampling.lower() == "healpix"
            else 1.0
        )

        for el in range(abs(spin), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))
            ftm[t, m_offset : 2 * L - 1 + m_offset] += (
                elfactor * dl * flm[el, :] * phase_shift
            )

    ftm *= (-1) ** (spin)

    if sampling.lower() == "healpix":
        f = hp.healpix_ifft(ftm, L, nside)
    else:
        f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def _compute_forward_direct(f, L, spin, sampling, thetas, weights, nside):
    r"""Compute forward spherical harmonic transform directly.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        weights (np.ndarray): Vector of quadrature weights on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

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


def _compute_forward_sov(f, L, spin, sampling, thetas, weights, nside):
    r"""Compute forward spherical harmonic transform by separation of variables with a
        manual Fourier transform.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        weights (np.ndarray): Vector of quadrature weights on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

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

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

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


def _compute_forward_sov_fft(f, L, spin, sampling, thetas, weights, nside):
    r"""Compute forward spherical harmonic transform by separation of variables with a
        Fast Fourier transform.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        weights (np.ndarray): Vector of quadrature weights on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside)
    else:
        ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):

        phi_ring_offset = (
            samples.p2phi_ring(t, 0, nside) if sampling.lower() == "healpix" else 0
        )

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
                        * phase_shift
                    )

    return flm


def _compute_forward_sov_fft_vectorized(f, L, spin, sampling, thetas, weights, nside):
    r"""A vectorized function to compute forward spherical harmonic transform by
        separation of variables with a manual Fourier transform.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.

        weights (np.ndarray): Vector of quadrature weights on the sphere.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside)
    else:
        ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    for t, theta in enumerate(thetas):

        phase_shift = (
            samples.ring_phase_shift_hp(L, t, nside, forward=True)
            if sampling.lower() == "healpix"
            else 1.0
        )

        for el in range(abs(spin), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            flm[el, :] += (
                weights[t]
                * elfactor
                * np.multiply(dl, ftm[t, m_offset : 2 * L - 1 + m_offset])
                * phase_shift
            )

    flm *= (-1) ** spin

    return flm

@partial(jit, static_argnums=(1, 2, 3, 6))
def _compute_forward_sov_fft_vectorized_jax_vmap(
    f, L, spin, sampling, thetas, weights, nside
):
    r"""A JAX version of the vectorized function to compute forward spherical harmonic transform by
        separation of variables with a manual Fourier transform.
    Args:
        f (np.ndarray): Signal on the sphere.
        L (int): Harmonic band-limit.
        spin (int): Harmonic spin.
        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.
        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.
        weights (np.ndarray): Vector of quadrature weights on the sphere.
        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix". 
    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # ftm array
    if sampling.lower() == "healpix":
        ftm = jnp.array(hp.healpix_fft_jax(f, L, nside, jnp))
    else:
        ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # Compute phase shift
    if sampling.lower() == "healpix":
        
        phase_shift_vmapped = jax.vmap(
            samples.ring_phase_shift_hp_vmappable, 
            in_axes=(None, 0, None, None), 
            out_axes=-1 # ATT! theta along last dim
        )  

        # expand to 3D (theta dim is last)
        phase_shift = phase_shift_vmapped(L, jnp.arange(len(thetas)), nside, True)[None,:,:]
        
    else:
        phase_shift = 1.0 #jnp.array(1.0)

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
        jnp.expand_dims(
            weights, axis=(0, 1)
        )  # Alternative to jnp.expand_dims: weights[None, None, :] --agnostic to np/jnp but seems slower?
        * jnp.expand_dims(
            jnp.sqrt(
                (2 * jnp.arange(abs(spin), L) + 1)
                / (4 * jnp.pi)
            ),
            axis=(-1, -2),
        )
        * dl_vmapped(thetas, jnp.arange(abs(spin), L), L, -spin)
        * jnp.expand_dims(
            jax.lax.slice_in_dim(ftm, m_offset, 2 * L - 1 + m_offset, axis=-1),
            axis=-1,
        ).T  # ftm[:, m_offset : 2 * L - 1 + m_offset, None].T
        * phase_shift  
    ).sum(axis=-1)

    flm *= (-1) ** spin

    # Pad the first n=spin rows with zeros
    flm = jnp.pad(flm, ((abs(spin), 0), (0, 0)))  # TODO: Do I need abs(spin)? check

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6))
def _compute_forward_sov_fft_vectorized_jax_map(
    f, L, spin, sampling, thetas, weights, nside
):
    r"""A JAX version of the vectorized function to compute forward spherical harmonic transform by
        separation of variables with a manual Fourier transform.
    Args:
        f (np.ndarray): Signal on the sphere.
        L (int): Harmonic band-limit.
        spin (int): Harmonic spin.
        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.
        thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.
        weights (np.ndarray): Vector of quadrature weights on the sphere.
        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix". 
    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # ftm array
    ftm = hp.healpix_fft_jax(f, L, nside, jnp) if sampling.lower() == "healpix" else (
        jnp.fft.fftshift(
            jnp.fft.fft(
                f, axis=1, norm="backward"), axes=1)
    )

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # phase shift
    phase_shifts = jnp.array([[1.]]) if sampling.lower() != "healpix" else jax.vmap(
        samples.ring_phase_shift_hp_vmappable, 
        in_axes=(None, 0, None, None),
        out_axes=-1
    )(L, jnp.arange(len(thetas)), nside, True)

    # el
    els = jnp.arange(abs(spin), L)
    elfactors = jnp.sqrt((2 * els + 1) / (4 * jnp.pi))

    # Compute flm
    flm =(
        weights[None, None, :] 
        * elfactors[:, None, None] 
        * jnp.moveaxis(
            jax.lax.map(
            lambda theta: jax.lax.map(
                lambda el: wigner.turok_jax.compute_slice(theta, el, L, -spin),
                els
            ),
            thetas
        ),
        0,-1)
        * ftm[:, m_offset : 2 * L - 1 + m_offset, None].T 
        * phase_shifts[None,:,:] 
    ).sum(axis=-1) * (-1) ** spin


    # Pad the first n=spin rows with zeros
    flm = jnp.pad(flm, ((abs(spin), 0), (0, 0)))  

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    return flm
