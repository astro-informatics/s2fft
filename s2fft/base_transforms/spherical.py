import numpy as np
from warnings import warn
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, resampling
from s2fft.utils import healpix_ffts as hp
from s2fft import recursions


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

    Uses a vectorised separation of variables method with np.fft.

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
    assert 0 <= L_lower < L

    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Deferring to complex transform."
        )

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

    Uses a vectorised separation of variables method with np.fft.

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
    assert 0 <= L_lower < L

    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Deferring to complex transform."
        )

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
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

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
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)
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
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            m_start_ind = 0 if reality else -el
            for m in range(m_start_ind, el + 1):
                phase_shift = (
                    np.exp(1j * m * phi_ring_offset)
                    if sampling.lower() == "healpix"
                    else 1
                )

                val = (
                    (-1) ** spin
                    * elfactor
                    * dl[m + L - 1]
                    * flm[el, m + L - 1]
                    * phase_shift
                )

                if reality and sampling.lower() == "healpix":
                    if m != 0:
                        ftm[t, -m + L - 1 + m_offset] += np.conj(val)

                ftm[t, m + L - 1 + m_offset] += val

    if sampling.lower() == "healpix":
        f = hp.healpix_ifft(ftm, L, nside, "numpy", reality)
    else:
        if reality:
            f = np.fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            f = np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

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
            samples.ring_phase_shift_hp(L, t, nside, False, reality)
            if sampling.lower() == "healpix"
            else 1.0
        )

        for el in range(max(L_lower, abs(spin)), L):
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)
            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))
            m_start_ind = L - 1 if reality else 0
            val = elfactor * dl[m_start_ind:] * flm[el, m_start_ind:] * phase_shift
            if reality and sampling.lower() == "healpix":
                ftm[t, m_offset : L - 1 + m_offset] += np.flip(np.conj(val[1:]))

            ftm[t, m_start_ind + m_offset : 2 * L - 1 + m_offset] += val

    ftm *= (-1) ** (spin)
    if sampling.lower() == "healpix":
        f = hp.healpix_ifft(ftm, L, nside, "numpy", reality)
    else:
        if reality:
            f = np.fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            f = np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")

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
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

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
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

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
        ftm = hp.healpix_fft(f, L, nside, "numpy", reality)
    else:
        if reality:
            ftm_temp = np.fft.rfft(
                np.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                ftm_temp = ftm_temp[:, :-1]
            ftm[:, L - 1 + m_offset :] = ftm_temp
        else:
            ftm = np.fft.fftshift(np.fft.fft(f, axis=1, norm="backward"), axes=1)

    for t, theta in enumerate(thetas):
        phi_ring_offset = (
            samples.p2phi_ring(t, 0, nside) if sampling.lower() == "healpix" else 0
        )

        for el in range(max(L_lower, abs(spin)), L):
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

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
        ftm = hp.healpix_fft(f, L, nside, "numpy", reality)
    else:
        if reality:
            t = np.fft.rfft(
                np.real(f),
                axis=1,
                norm="backward",
            )
            if m_offset != 0:
                t = t[:, :-1]
            ftm[:, L - 1 + m_offset :] = t
        else:
            ftm = np.fft.fftshift(np.fft.fft(f, axis=1, norm="backward"), axes=1)

    for t, theta in enumerate(thetas):
        phase_shift = (
            samples.ring_phase_shift_hp(L, t, nside, True, reality)
            if sampling.lower() == "healpix"
            else 1.0
        )

        for el in range(max(L_lower, abs(spin)), L):
            dl = recursions.turok.compute_slice(theta, el, L, -spin, reality)

            elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))

            m_start_ind = L - 1 if reality else 0
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
