from functools import partial
from warnings import warn

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
import numpy.fft as fft
from jax import jit

import s2fft.healpix_ffts as hp
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.samples as samples
import s2fft.wigner as wigner


def inverse(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Compute inverse spherical harmonic transform.

    Uses a vectorised separation of variables method with FFT.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    return _inverse(
        flm,
        L,
        spin,
        sampling,
        nside=nside,
        method="sov_fft_vectorized",
        reality=reality,
        L_lower=L_lower,
    )


def _inverse(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "sov_fft_vectorized",
    nside: int = None,
    reality: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Compute inverse spherical harmonic transform using a specified method.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Harmonic transform algorithm. Supported algorithms include
            {"direct", "sov", "sov_fft", "sov_fft_vectorized"}. Defaults to
            "sov_fft_vectorized".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    assert flm.shape == samples.flm_shape(L)
    assert L > 0
    assert 0 <= np.abs(spin) < L
    assert 0 <= L_lower < L

    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform."
        )

    if sampling.lower() == "healpix" and method not in ["direct", "sov"]:
        reality = False

    thetas = samples.thetas(L, sampling, nside)
    transform_methods = {
        "direct": _compute_inverse_direct,
        "sov": _compute_inverse_sov,
        "sov_fft": _compute_inverse_sov_fft,
        "sov_fft_vectorized": _compute_inverse_sov_fft_vectorized,
    }
    return transform_methods[method](
        flm,
        L,
        spin,
        sampling,
        thetas,
        nside=nside,
        reality=reality,
        L_lower=L_lower,
    )


def forward(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Compute forward spherical harmonic transform.

    Uses a vectorised separation of variables method with FFT.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    return _forward(
        f,
        L,
        spin,
        sampling,
        nside=nside,
        method="sov_fft_vectorized",
        reality=reality,
        L_lower=L_lower,
    )


def _forward(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "sov_fft_vectorized",
    nside: int = None,
    reality: bool = False,
    L_lower: int = 0,
):
    r"""Compute forward spherical harmonic transform using a specified method.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Harmonic transform algorithm. Supported algorithms include
            {"direct", "sov", "sov_fft", "sov_fft_vectorized"}. Defaults to
            "sov_fft_vectorized".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    assert f.shape == samples.f_shape(L, sampling, nside)
    assert L > 0
    assert 0 <= np.abs(spin) < L
    assert 0 <= L_lower < L

    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform."
        )

    if sampling.lower() == "healpix" and method not in ["direct", "sov"]:
        reality = False

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
        "jax_vmap_double": _compute_forward_jax_vmap_double,
        "jax_vmap_scan": _compute_forward_jax_vmap_scan,
        "jax_vmap_scan_0": _compute_forward_jax_vmap_scan_0,
        "jax_vmap_loop": _compute_forward_jax_vmap_loop,
        "jax_vmap_loop_0": _compute_forward_jax_vmap_loop_0,
        "jax_map": _compute_forward_jax_map,
    }
    return transform_methods[method](
        f,
        L,
        spin,
        sampling,
        thetas,
        weights,
        nside=nside,
        reality=reality,
        L_lower=L_lower,
    )


def _compute_inverse_direct(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    f = np.zeros(samples.f_shape(L, sampling, nside), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        if sampling.lower() == "healpix":
            phis_ring = samples.phis_ring(t, nside)

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            for p, phi in enumerate(phis_ring):

                if sampling.lower() != "healpix":
                    entry = (t, p)

                else:
                    entry = samples.hp_ang2pix(nside, theta, phi)

                if reality:
                    f[entry] += (
                        (-1) ** spin * elfactor * dl[L - 1] * flm[el, L - 1]
                    )  # m = 0
                    for m in range(1, el + 1):
                        val = (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1]
                            * flm[el, m + L - 1]
                        )
                        f[entry] += val + np.conj(val)

                else:
                    for m in range(-el, el + 1):
                        f[entry] += (
                            (-1) ** spin
                            * elfactor
                            * np.exp(1j * m * phi)
                            * dl[m + L - 1]
                            * flm[el, m + L - 1]
                        )

    return f


def _compute_inverse_sov(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    ftm = np.zeros((len(thetas), 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):
        for el in range(max(L_lower, abs(spin)), L):
            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            m_start_ind = 0 if reality else -el
            for m in range(m_start_ind, el + 1):
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
            if sampling.lower() != "healpix":
                entry = (t, p)
            else:
                entry = samples.hp_ang2pix(nside, theta, phi)
            if reality:
                f[entry] += ftm[t, L - 1]  # m = 0
                for m in range(1, L):
                    val = ftm[t, m + L - 1] * np.exp(1j * m * phi)
                    f[entry] += val + np.conj(val)
            else:
                for m in range(-(L - 1), L):
                    f[entry] += ftm[t, m + L - 1] * np.exp(1j * m * phi)

    return f


def _compute_inverse_sov_fft(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

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

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            m_start_ind = 0 if reality else -el
            for m in range(m_start_ind, el + 1):

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
        if reality:
            f = fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def _compute_inverse_sov_fft_vectorized(
    flm: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

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

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))
            m_start_ind = L - 1 if reality else 0
            ftm[t, m_start_ind + m_offset : 2 * L - 1 + m_offset] += (
                elfactor * dl[m_start_ind:] * flm[el, m_start_ind:] * phase_shift
            )

    ftm *= (-1) ** (spin)
    if sampling.lower() == "healpix":
        f = hp.healpix_ifft(ftm, L, nside)
    else:
        if reality:
            f = fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            f = fft.ifft(fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

    return f


def _compute_forward_direct(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
):
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    for t, theta in enumerate(thetas):

        if sampling.lower() == "healpix":
            phis_ring = samples.phis_ring(t, nside)

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            for p, phi in enumerate(phis_ring):

                if sampling.lower() != "healpix":
                    entry = (t, p)
                else:
                    entry = samples.hp_ang2pix(nside, theta, phi)

                if reality:
                    flm[el, L - 1] += (
                        weights[t] * (-1) ** spin * elfactor * dl[L - 1] * f[entry]
                    )  # m = 0
                    for m in range(1, el + 1):
                        val = (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1]
                            * f[entry]
                        )
                        flm[el, m + L - 1] += val
                        flm[el, -m + L - 1] += (-1) ** m * np.conj(val)

                else:
                    for m in range(-el, el + 1):
                        flm[el, m + L - 1] += (
                            weights[t]
                            * (-1) ** spin
                            * elfactor
                            * np.exp(-1j * m * phi)
                            * dl[m + L - 1]
                            * f[entry]
                        )

    return flm


def _compute_forward_sov(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
):
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    if sampling.lower() != "healpix":
        phis_ring = samples.phis_equiang(L, sampling)

    ftm = np.zeros((len(thetas), 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):

        if sampling.lower() == "healpix":
            phis_ring = samples.phis_ring(t, nside)

        for p, phi in enumerate(phis_ring):
            if sampling.lower() != "healpix":
                entry = (t, p)
            else:
                entry = samples.hp_ang2pix(nside, theta, phi)

            m_start_ind = 0 if reality else -L + 1
            for m in range(m_start_ind, L):
                ftm[t, m + L - 1] += np.exp(-1j * m * phi) * f[entry]

    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    for t, theta in enumerate(thetas):

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            if reality:
                flm[el, L - 1] += (
                    weights[t] * (-1) ** spin * elfactor * dl[L - 1] * ftm[t, L - 1]
                )  # m = 0
                for m in range(1, el + 1):
                    val = (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1]
                    )
                    flm[el, m + L - 1] += val
                    flm[el, -m + L - 1] += (-1) ** m * np.conj(val)

            else:
                for m in range(-el, el + 1):
                    flm[el, m + L - 1] += (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1]
                    )

    return flm


def _compute_forward_sov_fft(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
):
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    ftm = np.zeros_like(f).astype(np.complex128)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside)
    else:
        if reality:
            ftm_temp = fft.rfft(
                np.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                ftm_temp = ftm_temp[:, :-1]
            ftm[:, L - 1 + m_offset :] = ftm_temp
        else:
            ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    for t, theta in enumerate(thetas):

        phi_ring_offset = (
            samples.p2phi_ring(t, 0, nside) if sampling.lower() == "healpix" else 0
        )

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            if reality:
                flm[el, L - 1] += (
                    weights[t]
                    * (-1) ** spin
                    * elfactor
                    * dl[L - 1]
                    * ftm[t, L - 1 + m_offset]
                )
                for m in range(1, el + 1):
                    phase_shift = (
                        np.exp(-1j * m * phi_ring_offset)
                        if sampling.lower() == "healpix"
                        else 1
                    )

                    val = (
                        weights[t]
                        * (-1) ** spin
                        * elfactor
                        * dl[m + L - 1]
                        * ftm[t, m + L - 1 + m_offset]
                        * phase_shift
                    )
                    flm[el, m + L - 1] += val
                    flm[el, -m + L - 1] += (-1) ** m * np.conj(val)
            else:
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


def _compute_forward_sov_fft_vectorized(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
):
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    ftm = np.zeros_like(f).astype(np.complex128)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if reality:
        m_conj = (-1) ** (np.arange(1, L) % 2)

    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside)
    else:
        if reality:
            t = fft.rfft(
                np.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]
            ftm[:, L - 1 + m_offset :] = t
        else:
            ftm = fft.fftshift(fft.fft(f, axis=1, norm="backward"), axes=1)

    m_start_ind = L - 1 if reality else 0

    for t, theta in enumerate(thetas):

        phase_shift = (
            samples.ring_phase_shift_hp(L, t, nside, forward=True)
            if sampling.lower() == "healpix"
            else 1.0
        )

        for el in range(max(L_lower, abs(spin)), L):

            dl = wigner.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            flm[el, m_start_ind:] += (
                weights[t]
                * elfactor
                * np.multiply(
                    dl[m_start_ind:],
                    ftm[t, m_start_ind + m_offset : 2 * L - 1 + m_offset],
                )
                * phase_shift
            )
            if reality:
                flm[el, :m_start_ind] = np.flip(
                    m_conj * np.conj(flm[el, m_start_ind + 1 :])
                )

    flm *= (-1) ** spin

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_vmap_double(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # phase shift
    if sampling.lower() == "healpix":

        phase_shift_vmapped = jax.vmap(
            samples.ring_phase_shift_hp_vmappable,
            in_axes=(None, 0, None, None),
            out_axes=-1,  # theta along last dimensio
        )

        # expand to 3D array (theta dim is the last)
        phase_shift = phase_shift_vmapped(L, jnp.arange(len(thetas)), nside, True)[
            None, :, :
        ]

    else:
        phase_shift = 1.0

    # m start index
    m_start_ind = L - 1 if reality else 0

    # Compute dl_vmapped fn (double vmap)
    dl_vmapped = jax.vmap(
        jax.vmap(
            wigner.turok_jax.compute_slice,
            in_axes=(0, None, None, None, None),
            out_axes=-1,
        ),
        in_axes=(None, 0, None, None, None),
        out_axes=0,
    )

    # Compute flm
    el_array = jnp.arange(max(L_lower, abs(spin)), L)
    flm = jnp.zeros(
        (len(el_array), samples.flm_shape(L)[-1], len(thetas)), dtype=jnp.complex128
    )

    flm = flm.at[:, m_start_ind:, :].set(
        weights[None, None, :]
        * jnp.sqrt((2 * el_array + 1) / (4 * jnp.pi))[:, None, None]
        * dl_vmapped(thetas, el_array, L, -spin, reality)[:, m_start_ind:, :]
        * jax.lax.slice_in_dim(
            ftm, m_start_ind + m_offset, 2 * L - 1 + m_offset, axis=-1
        )[:, :, None].T
        * phase_shift
    )
    flm = flm.sum(axis=-1)

    # Pad the first n=max(L_lower, abs(spin)) rows with zeros
    flm = jnp.pad(flm, ((max(L_lower, abs(spin)), 0), (0, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # Apply spin
    flm *= (-1) ** spin

    # if reality=True: fill the first half of the columns w/ conjugate symmetric values
    if reality:
        # m conj
        m_conj = (-1) ** (jnp.arange(1, L) % 2)

        flm = flm.at[:, :m_start_ind].set(
            jnp.flip(m_conj * jnp.conj(flm[:, m_start_ind + 1 :]), axis=-1)
        )

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_vmap_scan(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # # phase shift
    if sampling.lower() == "healpix":

        phase_shift_vmapped = jax.vmap(
            samples.ring_phase_shift_hp_vmappable,
            in_axes=(None, 0, None, None),
            out_axes=0,  # theta along first dimension
        )
        phase_shift = phase_shift_vmapped(L, jnp.arange(len(thetas)), nside, True)

    else:
        phase_shift = jnp.ones_like(thetas)

    # Compute dl_vmapped fn (single vmap)
    dl_vmapped = jax.vmap(
        wigner.turok_jax.compute_slice,
        in_axes=(None, 0, None, None, None),  # map along el
        out_axes=0,  # place in first dim
    )
    # m start index
    if reality:
        m_start_ind = L - 1
        m_conj = (-1) ** (jnp.arange(1, L) % 2)
    else:
        m_start_ind = 0

    # initialise flm_half (second half of the columns, from m_start_ind)
    el_array = jnp.arange(max(L_lower, abs(spin)), L)
    flm_half = jnp.zeros(
        (len(el_array), samples.flm_shape(L)[-1] - m_start_ind),
        dtype=jnp.complex128,  # len(thetas)),
    )

    ###
    # compute flm using lax.scan over theta
    def cumsum(flm_half_carry, theta_weight_ftm_phase_shift):
        theta, weight, ftm_one, phase_shift_one = theta_weight_ftm_phase_shift
        flm_half_carry += (
            weight
            * jnp.sqrt((2 * el_array + 1) / (4 * jnp.pi))[:, None]
            * dl_vmapped(theta, el_array, L, -spin, reality)[:, m_start_ind:]
            * jax.lax.slice_in_dim(
                ftm_one, m_start_ind + m_offset, 2 * L - 1 + m_offset, axis=0
            )[:, None].T
            * phase_shift_one
        )
        return flm_half_carry, flm_half_carry

    theta_weight_ftm_phase_shift = (thetas, weights, ftm, phase_shift)
    flm_half, _ = jax.lax.scan(cumsum, flm_half, theta_weight_ftm_phase_shift)

    # Compute full flm
    # 1- Pad the first n=max(L_lower, abs(spin)) rows with zeros,
    flm_half = jnp.pad(flm_half, ((max(L_lower, abs(spin)), 0), (0, 0)))
    # 2- if reality True: concatenate flm_half_padded with symm conj along columns
    #    if reality False: pad the first m_start_ind columns w zeros
    if reality:
        flm = jnp.hstack(
            (
                jnp.flip(m_conj * jnp.conj(flm_half[:, 1:]), axis=-1),
                flm_half,
            )
        )
    else:
        flm = jnp.pad(flm_half, ((0, 0), (m_start_ind, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # Apply spin
    flm *= (-1) ** spin

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_vmap_scan_0(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # # phase shift
    if sampling.lower() == "healpix":

        phase_shift_vmapped = jax.vmap(
            samples.ring_phase_shift_hp_vmappable,
            in_axes=(None, 0, None, None),
            out_axes=0,  # theta along first dimension
        )
        phase_shift = phase_shift_vmapped(L, jnp.arange(len(thetas)), nside, True)

    else:
        phase_shift = jnp.ones_like(thetas)

    # Compute dl_vmapped fn (single vmap)
    dl_vmapped = jax.vmap(
        wigner.turok_jax.compute_slice,
        in_axes=(None, 0, None, None, None),  # map along el
        out_axes=0,  # place in first dim
    )
    # m start index
    m_start_ind = L - 1 if reality else 0

    # initialise flm
    el_array = jnp.arange(max(L_lower, abs(spin)), L)
    flm = jnp.zeros(
        (len(el_array), samples.flm_shape(L)[-1]), dtype=jnp.complex128  # len(thetas)),
    )

    ###
    # compute flm using lax.scan over theta
    def cumsum(flm_carry, theta_weight_ftm_phase_shift):
        theta, weight, ftm_one, phase_shift_one = theta_weight_ftm_phase_shift
        flm_carry = flm_carry.at[:, m_start_ind:].add(
            weight
            * jnp.sqrt((2 * el_array + 1) / (4 * jnp.pi))[:, None]
            * dl_vmapped(theta, el_array, L, -spin, reality)[:, m_start_ind:]
            * jax.lax.slice_in_dim(
                ftm_one, m_start_ind + m_offset, 2 * L - 1 + m_offset, axis=0
            )[:, None].T
            * phase_shift_one
        )
        return flm_carry, flm_carry

    theta_weight_ftm_phase_shift = (thetas, weights, ftm, phase_shift)
    flm, _ = jax.lax.scan(cumsum, flm, theta_weight_ftm_phase_shift)

    # Pad the first n=max(L_lower, abs(spin)) rows with zeros
    flm = jnp.pad(flm, ((max(L_lower, abs(spin)), 0), (0, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # Apply spin
    flm *= (-1) ** spin

    # if reality=True: fill the first half of the columns w/ conjugate symmetric values
    if reality:
        # m conj
        m_conj = (-1) ** (jnp.arange(1, L) % 2)

        flm = flm.at[:, :m_start_ind].set(
            jnp.flip(m_conj * jnp.conj(flm[:, m_start_ind + 1 :]), axis=-1)
        )

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_vmap_loop(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # Compute dl_vmapped fn (single vmap)
    dl_vmapped = jax.vmap(
        wigner.turok_jax.compute_slice,
        in_axes=(None, 0, None, None, None),  # map along el
        out_axes=0,  # place in first dim
    )

    # phase shift
    phase_shift = (
        samples.ring_phase_shift_hp_vmappable(L, ti, nside, True)
        if sampling.lower() == "healpix"
        else 1.0
    )

    # m start index
    if reality:
        m_start_ind = L - 1
        m_conj = (-1) ** (jnp.arange(1, L) % 2)
    else:
        m_start_ind = 0

    # initialise flm_half (second half of the columns, from m_start_ind)
    el_array = jnp.arange(max(L_lower, abs(spin)), L)
    flm_half = jnp.zeros(
        (len(el_array), samples.flm_shape(L)[-1] - m_start_ind),
        dtype=jnp.complex128,  # len(thetas)),
    )

    ###
    # compute flm_half using a manual loop over theta
    for ti, theta in enumerate(thetas):  # ti not traced
        flm_half += (
            weights[ti]
            * jnp.sqrt((2 * el_array + 1) / (4 * jnp.pi))[:, None]
            * dl_vmapped(theta, el_array, L, -spin, reality)[:, m_start_ind:]
            * jax.lax.slice_in_dim(
                ftm[ti, :], m_start_ind + m_offset, 2 * L - 1 + m_offset, axis=0
            )[:, None].T
            * phase_shift
        )

    # Compute full flm
    # 1- Pad the first n=max(L_lower, abs(spin)) rows with zeros,
    flm_half = jnp.pad(flm_half, ((max(L_lower, abs(spin)), 0), (0, 0)))
    # 2- if reality True: concatenate flm_half_padded with symm conj along columns
    #    if reality False: pad the first m_start_ind columns w zeros
    if reality:
        flm = jnp.hstack(
            (
                jnp.flip(m_conj * jnp.conj(flm_half[:, 1:]), axis=-1),
                flm_half,
            )
        )
    else:
        flm = jnp.pad(flm_half, ((0, 0), (m_start_ind, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # Apply spin
    flm *= (-1) ** spin

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_vmap_loop_0(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
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
            if sampling="healpix".  Defaults to None.
        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
        L_lower (int): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`.  Defaults to 0.
    Returns:
        np.ndarray: Spherical harmonic coefficients.
    """

    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # Compute dl_vmapped fn (single vmap)
    dl_vmapped = jax.vmap(
        wigner.turok_jax.compute_slice,
        in_axes=(None, 0, None, None, None),  # map along el
        out_axes=0,  # place in first dim
    )
    # phase shift
    phase_shift = (
        samples.ring_phase_shift_hp_vmappable(L, ti, nside, True)
        if sampling.lower() == "healpix"
        else 1.0
    )

    # m start index
    if reality:
        m_start_ind = L - 1
        m_conj = (-1) ** (jnp.arange(1, L) % 2)
    else:
        m_start_ind = 0

    # initialise flm
    el_array = jnp.arange(max(L_lower, abs(spin)), L)
    flm = jnp.zeros((len(el_array), samples.flm_shape(L)[-1]), dtype=jnp.complex128)

    ###
    # compute flm using a manual loop over theta
    for ti, theta in enumerate(thetas):
        flm = flm.at[:, m_start_ind:].add(
            weights[ti]
            * jnp.sqrt((2 * el_array + 1) / (4 * jnp.pi))[:, None]
            * dl_vmapped(theta, el_array, L, -spin, reality)[:, m_start_ind:]
            * jax.lax.slice_in_dim(
                ftm[ti, :], m_start_ind + m_offset, 2 * L - 1 + m_offset, axis=0
            )[:, None].T
            * phase_shift
        )

    # Pad the first n=max(L_lower, abs(spin)) rows with zeros
    flm = jnp.pad(flm, ((max(L_lower, abs(spin)), 0), (0, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # Apply spin
    flm *= (-1) ** spin

    # if reality=True: fill the first half of the columns w/ conjugate symmetric values
    if reality:
        flm = flm.at[:, :m_start_ind].set(
            jnp.flip(m_conj * jnp.conj(flm[:, m_start_ind + 1 :]), axis=-1)
        )

    return flm


@partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def _compute_forward_jax_map(
    f: np.ndarray,
    L: int,
    spin: int,
    sampling: str,
    thetas: np.ndarray,
    weights: np.ndarray,
    nside: int,
    reality: bool,
    L_lower: int,
):
    # m offset
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    # ftm array
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft_jax(f, L, nside, jnp)
    else:
        if reality:
            t = jfft.rfft(
                jnp.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]

            # build ftm by concatenating t with a zero array
            ftm = jnp.hstack(
                (jnp.zeros((f.shape[0], L - 1 + m_offset)).astype(jnp.complex128), t),
            )
        else:
            ftm = jfft.fftshift(jfft.fft(f, axis=1, norm="backward"), axes=1)

    # phase shift
    if sampling.lower() != "healpix":
        phase_shifts = jnp.array([[1.0]])
    else:
        phase_shifts = jax.vmap(
            samples.ring_phase_shift_hp_vmappable, in_axes=(None, 0, None, None)
        )(L, jnp.arange(len(thetas)), nside, True)

    # m start index
    m_start_ind = L - 1 if reality else 0

    # Compute flm
    els = jnp.arange(max(L_lower, abs(spin)), L)
    elfactors = jnp.sqrt((2 * els + 1) / (4 * jnp.pi))
    flm = jnp.zeros(
        (len(thetas), len(els), samples.flm_shape(L)[-1]), dtype=jnp.complex128
    )

    # Compute dl (double map)
    dl_mapped = jax.lax.map(
        lambda theta: jax.lax.map(
            lambda el: wigner.turok_jax.compute_slice(theta, el, L, -spin), els
        ),
        thetas,
    )

    flm = (
        flm.at[:, :, m_start_ind:]
        .set(
            weights[:, None, None]
            * elfactors[None, :, None]
            * dl_mapped[:, :, m_start_ind:]
            * ftm[:, None, m_start_ind + m_offset : 2 * L - 1 + m_offset]
            * phase_shifts[:, None, :]
        )
        .sum(axis=0)
    )

    # Apply spin
    flm *= (-1) ** spin

    # Pad the first n=max(L_lower, abs(spin)) rows with zeros
    flm = jnp.pad(flm, ((max(L_lower, abs(spin)), 0), (0, 0)))

    # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
    upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
    mask = upper_diag * jnp.fliplr(upper_diag)
    flm *= mask

    # if reality=True: fill the first half of the columns w/ conjugate symmetric values
    if reality:
        # m conj
        m_conj = (-1) ** (jnp.arange(1, L) % 2)

        flm = flm.at[:, :m_start_ind].set(
            jnp.flip(m_conj * jnp.conj(flm[:, m_start_ind + 1 :]), axis=-1)
        )

    return flm

################
# @partial(jit, static_argnums=(1, 2, 3, 6))
# def _compute_forward_sov_fft_vectorized_jax_map(
#     f, L, spin, sampling, thetas, weights, nside
# ):
#     r"""A JAX version of the vectorized function to compute forward spherical harmonic transform by
#         separation of variables with a manual Fourier transform.
#     Args:
#         f (np.ndarray): Signal on the sphere.
#         L (int): Harmonic band-limit.
#         spin (int): Harmonic spin.
#         sampling (str): Sampling scheme.  Supported sampling schemes include
#             {"mw", "mwss", "dh", "healpix"}.
#         thetas (np.ndarray): Vector of sample positions in :math:`\theta` on the sphere.
#         weights (np.ndarray): Vector of quadrature weights on the sphere.
#         nside (int): HEALPix Nside resolution parameter.  Only required
#             if sampling="healpix".
#     Returns:
#         np.ndarray: Spherical harmonic coefficients.
#     """

#     # ftm array
#     ftm = (
#         hp.healpix_fft_jax(f, L, nside, jnp)
#         if sampling.lower() == "healpix"
#         else (jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1))
#     )

#     # m offset
#     m_offset = 1 if sampling in ["mwss", "healpix"] else 0

#     # phase shift
#     phase_shifts = (
#         jnp.array([[1.0]])
#         if sampling.lower() != "healpix"
#         else jax.vmap(
#             samples.ring_phase_shift_hp_vmappable,
#             in_axes=(None, 0, None, None),
#             out_axes=-1,
#         )(L, jnp.arange(len(thetas)), nside, True)
#     )

#     # el
#     els = jnp.arange(abs(spin), L)
#     elfactors = jnp.sqrt((2 * els + 1) / (4 * jnp.pi))

#     # Compute flm
#     flm = (
#         weights[None, None, :]
#         * elfactors[:, None, None]
#         * jnp.moveaxis(
#             jax.lax.map(
#                 lambda theta: jax.lax.map(
#                     lambda el: wigner.turok_jax.compute_slice(theta, el, L, -spin), els
#                 ),
#                 thetas,
#             ),
#             0,
#             -1,
#         )
#         * ftm[:, m_offset : 2 * L - 1 + m_offset, None].T
#         * phase_shifts[None, :, :]
#     ).sum(axis=-1) * (-1) ** spin

#     # Pad the first n=spin rows with zeros
#     flm = jnp.pad(flm, ((abs(spin), 0), (0, 0)))

#     # Mask after pad (to set spurious results from wigner.turok_jax.compute_slice to zero)
#     upper_diag = jnp.triu(jnp.ones_like(flm, dtype=bool).T, k=-(L - 1)).T
#     mask = upper_diag * jnp.fliplr(upper_diag)
#     flm *= mask

#     return flm
