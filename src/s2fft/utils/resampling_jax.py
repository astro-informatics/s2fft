from functools import partial as _partial

import jax.numpy as jnp
from jax import jit as _jit

from s2fft.sampling import s2_samples as samples
from s2fft.utils._dtype_association import compatible_complex_dtype


@_partial(_jit, static_argnums=(1))
def mw_to_mwss(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    r"""
    Convert signal on the sphere from MW sampling to MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.  Second, zero padding in harmonic space corresponding to
    :math:`\phi` is performed.

    JAX implementation of :func:`~s2fft.resampling.mw_to_mwss`.

    Args:
        f_mw (jnp.ndarray): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Signal on the sphere sampled with MWSS sampling.

    """
    if f_mw.ndim == 2:
        return jnp.squeeze(
            mw_to_mwss_phi(mw_to_mwss_theta(jnp.expand_dims(f_mw, 0), L, spin), L)
        )
    else:
        return mw_to_mwss_phi(mw_to_mwss_theta(f_mw, L, spin), L)


@_partial(_jit, static_argnums=(1))
def mw_to_mwss_theta(f_mw: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    r"""
    Convert :math:`\theta` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by first performing a period extension in
    :math:`\theta` to :math:`2\pi`, followed by zero padding in harmonic space.  The
    resulting signal is then unextend back to the :math:`\theta` domain of
    :math:`[0,\pi]`.

    JAX implementation of :func:`~s2fft.resampling.mw_to_mwss_theta`.


    Args:
        f_mw (jnp.ndarray): Signal on the sphere sampled with MW sampling.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Raises:
        ValueError: Input spherical signal must have shape matching MW sampling.

    Returns:
        jnp.ndarray: Signal on the sphere with MWSS sampling in :math:`\theta` and MW
        sampling in :math:`\phi`.

    """
    f_mw_ext = periodic_extension(f_mw, L, spin=spin, sampling="mw")
    fmp_mwss_ext = jnp.zeros(
        (f_mw_ext.shape[0], 2 * L, 2 * L - 1),
        dtype=compatible_complex_dtype(f_mw),
    )

    fmp_mwss_ext = fmp_mwss_ext.at[:, 1:, :].set(
        jnp.fft.fftshift(jnp.fft.fft(f_mw_ext, axis=-2, norm="forward"), axes=-2)
    )

    fmp_mwss_ext = fmp_mwss_ext.at[:, 1:, :].set(
        jnp.einsum(
            "...blp,l->...blp",
            fmp_mwss_ext[:, 1:, :],
            jnp.exp(
                -1j
                * jnp.arange(-(L - 1), L, dtype=compatible_complex_dtype(f_mw))
                * jnp.pi
                / (2 * L - 1)
            ),
            optimize=True,
        )
    )

    f_mwss_ext = jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(fmp_mwss_ext), axes=-2),
            axis=-2,
            norm="backward",
        )
    )

    return unextend(f_mwss_ext, L, sampling="mwss")


@_partial(_jit, static_argnums=(1))
def mw_to_mwss_phi(f_mw: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Convert :math:`\phi` component of signal on the sphere from MW sampling to
    MWSS sampling.

    Conversion is performed by zero padding in harmonic space.

    JAX implementation of :func:`~s2fft.resampling.mw_to_mwss_phi`.


    Note:
        Can work with arbitrary number of :math:`\theta` samples.  Hence, to convert
        both :math:`(\theta,\phi)` sampling to MWSS, can use :func:`~mw_to_mwss_theta`
        to first convert :math:`\theta` sampling before using this function to convert
        the :math:`\phi` sampling.

    Args:
        f_mw (jnp.ndarray): Signal on the sphere sampled with MW sampling in
            :math:`\phi` and arbitrary number of samples in

        L (int): Harmonic band-limit.

    Raises:
        ValueError: Input spherical signal must have number of samples in :math:`\phi`
            matching MW sampling.

    Returns:
        jnp.ndarray: Signal on the sphere with MWSS sampling in :math:`\phi` and
        sampling in :math:`\theta` of the input signal.

    """
    f_mwss = jnp.zeros(
        (f_mw.shape[0], L + 1, 2 * L), dtype=compatible_complex_dtype(f_mw)
    )
    f_mwss = f_mwss.at[:, :, 1:].set(
        jnp.fft.fftshift(jnp.fft.fft(f_mw, axis=-1, norm="forward"), axes=-1)
    )

    return jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(f_mwss), axes=-1),
            axis=-1,
            norm="backward",
        )
    )


@_partial(_jit, static_argnums=(1, 3))
def periodic_extension(
    f: jnp.ndarray, L: int, spin: int = 0, sampling: str = "mw"
) -> jnp.ndarray:
    r"""
    Perform period extension of MW/MWSS signal on the sphere in harmonic
    domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
    JAX implementation of :func:`~s2fft.resampling.periodic_extension`.

    Args:
        f (jnp.ndarray): Signal on the sphere sampled with MW/MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

    Returns:
        jnp.ndarray: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in same scheme (MW/MWSS) as input.

    """
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ntheta_ext = samples.ntheta_extension(L, sampling)
    m_offset = 1 if sampling == "mwss" else 0

    f_ext = jnp.zeros((f.shape[0], ntheta_ext, nphi), dtype=compatible_complex_dtype(f))
    f_ext = f_ext.at[:, 0:ntheta, 0:nphi].set(f[:, 0:ntheta, 0:nphi])
    f_ext = jnp.fft.fftshift(jnp.fft.fft(f_ext, axis=-1, norm="backward"), axes=-1)

    f_ext = f_ext.at[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ].set(
        jnp.flip(
            f_ext[
                :,
                m_offset : L - 1 + m_offset,
                m_offset : 2 * L - 1 + m_offset,
            ],
            axis=-2,
        )
    )
    f_ext = f_ext.at[
        :,
        L + m_offset : 2 * L - 1 + m_offset,
        m_offset : 2 * L - 1 + m_offset,
    ].multiply((-1) ** (jnp.arange(-(L - 1), L)))
    if spin.size > 1:
        f_ext = f_ext.at[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ].set(
            jnp.einsum(
                "nlm,n->nlm",
                f_ext[
                    :,
                    L + m_offset : 2 * L - 1 + m_offset,
                    m_offset : 2 * L - 1 + m_offset,
                ],
                (-1) ** spin,
                optimize=True,
            )
        )
    else:
        f_ext = f_ext.at[
            :,
            L + m_offset : 2 * L - 1 + m_offset,
            m_offset : 2 * L - 1 + m_offset,
        ].multiply((-1) ** spin)

    return (
        jnp.conj(
            jnp.fft.fft(
                jnp.fft.ifftshift(jnp.conj(f_ext), axes=-1),
                axis=-1,
                norm="backward",
            )
        )
        / nphi
    )


@_partial(_jit, static_argnums=(2,))
def periodic_extension_cc_f1(
    ftm: jnp.ndarray, spin: int = 0, sampling: str = "cc"
) -> jnp.ndarray:
    """Continue longitude Fourier coefficients over a full meridian."""
    sampling = sampling.lower()
    if sampling not in ("cc", "f1"):
        raise ValueError("Meridional continuation only supports cc and f1")
    n_phi = ftm.shape[-1]
    modes = jnp.rint(jnp.fft.fftshift(jnp.fft.fftfreq(n_phi) * n_phi)).astype(jnp.int32)
    parity = (-1.0) ** (modes + spin)
    reflected = (
        jnp.flip(ftm, axis=-2)
        if sampling == "f1"
        else jnp.flip(ftm[..., 1:-1, :], axis=-2)
    )
    return jnp.concatenate(
        [ftm, reflected * parity.reshape((1,) * (ftm.ndim - 1) + (n_phi,))],
        axis=-2,
    )


@_partial(_jit, static_argnums=(1,))
def _upsample_periodic_fourier(
    f: jnp.ndarray, theta_offset: float = 0.0
) -> jnp.ndarray:
    """Double a periodic meridian and evaluate it on a grid starting at zero."""
    n_theta = f.shape[-2]
    n_theta_up = 2 * n_theta
    raw_spectrum = jnp.fft.fft(f, axis=-2, norm="forward")
    spectrum_up = jnp.zeros(
        f.shape[:-2] + (n_theta_up, f.shape[-1]), dtype=raw_spectrum.dtype
    )
    n_half = n_theta // 2
    modes = jnp.arange(n_theta)
    modes = jnp.where(modes < n_half, modes, modes - n_theta)
    phase = jnp.exp(-1j * modes * theta_offset)
    spectrum = raw_spectrum * phase.reshape((1,) * (f.ndim - 2) + (n_theta, 1))
    spectrum_up = spectrum_up.at[..., :n_half, :].set(spectrum[..., :n_half, :])
    spectrum_up = spectrum_up.at[..., n_theta_up - (n_half - 1) :, :].set(
        spectrum[..., n_half + 1 :, :]
    )

    nyquist = raw_spectrum[..., n_half, :]
    nyquist_minus = 0.5 * nyquist * jnp.exp(1j * n_half * theta_offset)
    nyquist_plus = 0.5 * nyquist * jnp.exp(-1j * n_half * theta_offset)
    spectrum_up = spectrum_up.at[..., n_half, :].set(nyquist_plus)
    spectrum_up = spectrum_up.at[..., n_theta_up - n_half, :].set(nyquist_minus)
    return jnp.fft.ifft(spectrum_up, axis=-2, norm="forward")


@_partial(_jit, static_argnums=(2,))
def resample_cc_f1_for_analysis(
    ftm: jnp.ndarray, spin: int, sampling: str
) -> jnp.ndarray:
    """Resample CC/F1 Fourier coefficients to denser CC latitude nodes."""
    sampling = sampling.lower()
    ftm_ext = periodic_extension_cc_f1(ftm, spin, sampling)
    if sampling == "cc":
        theta_offset = 0.0
        n_dense = 2 * ftm.shape[-2] - 1
    elif sampling == "f1":
        theta_offset = jnp.pi / ftm_ext.shape[-2]
        n_dense = 2 * ftm.shape[-2] + 1
    else:
        raise ValueError("Resampling only supports cc and f1")
    ftm_dense = _upsample_periodic_fourier(ftm_ext, theta_offset)
    return ftm_dense[..., :n_dense, :]


@_partial(_jit, static_argnums=(1, 2))
def unextend(f_ext: jnp.ndarray, L: int, sampling: str = "mw") -> jnp.ndarray:
    r"""
    Unextend MW/MWSS sampled signal from :math:`\theta` domain
    :math:`[0,2\pi]` to :math:`[0,\pi]`.

    Args:
        f_ext (jnp.ndarray): Signal on the sphere sampled on extended :math:`\theta`
            domain :math:`[0,2\pi]`.

        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss"}.  Defaults to "mw".

    Raises:
        ValueError: Only MW/MWW sampling schemes supported.

        ValueError: Period extension must have correct shape.

    Returns:
        jnp.ndarray: Signal on the sphere sampled on :math:`\theta` domain
        :math:`[0,\pi]`.

    """
    if sampling.lower() == "mw":
        return f_ext[:, 0:L, :]

    elif sampling.lower() == "mwss":
        return f_ext[:, 0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )


@_partial(_jit, static_argnums=(1))
def upsample_by_two_mwss(f: jnp.ndarray, L: int, spin: int = 0) -> jnp.ndarray:
    r"""
    Upsample MWSS sampled signal on the sphere defined on domain :math:`[0,\pi]`
    by a factor of two.

    Upsampling is performed by a periodic extension in :math:`\theta` to
    :math:`[0,2\pi]`, followed by zero-padding in harmonic space, followed by
    unextending :math:`\theta` domain back to :math:`[0,\pi]`.

    JAX implementation of :func:`~s2fft.resampling.upsample_by_two_mwss`.

    Args:
        f (jnp.ndarray): Signal on the sphere sampled with MWSS sampling scheme, sampled
            at resolution L.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Signal on the sphere sampled with MWSS sampling scheme, sampling at
        resolution 2*L.

    """
    if f.ndim == 2:
        f = jnp.expand_dims(f, 0)
    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext = upsample_by_two_mwss_ext(f_ext, L)
    f_ext = unextend(f_ext, 2 * L, sampling="mwss")
    return jnp.squeeze(f_ext)


@_partial(_jit, static_argnums=(1))
def upsample_by_two_mwss_ext(f_ext: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Upsample an extended MWSS sampled signal on the sphere defined on domain
    :math:`[0,2\pi]` by a factor of two.

    Upsampling is performed by zero-padding in harmonic space. JAX implementation of
    :func:`~s2fft.resampling.upsample_by_two_mwss_ext`.

    Args:
        f_ext (jnp.ndarray): Signal on the sphere sampled on extended MWSS sampling
            scheme on domain :math:`[0,2\pi]`, sampled at resolution L.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Signal on the sphere sampled on extended MWSS sampling scheme on
        domain :math:`[0,2\pi]`, sampling at resolution 2*L.

    """
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.fft.fftshift(jnp.fft.fft(f_ext, axis=-2, norm="forward"), axes=-2)

    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = jnp.zeros(
        (f_ext.shape[0], ntheta_ext_up, nphi),
        dtype=compatible_complex_dtype(f_ext),
    )
    f_ext_up = f_ext_up.at[:, L : ntheta_ext + L, :nphi].set(
        f_ext[:, 0:ntheta_ext, :nphi]
    )
    return jnp.conj(
        jnp.fft.fft(
            jnp.fft.ifftshift(jnp.conj(f_ext_up), axes=-2),
            axis=-2,
            norm="backward",
        )
    )


@_partial(_jit, static_argnums=(1))
def periodic_extension_spatial_mwss(
    f: jnp.ndarray, L: int, spin: int = 0
) -> jnp.ndarray:
    r"""
    Perform period extension of MWSS signal on the sphere in spatial domain,
    extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.

    For the MWSS sampling scheme, it is possible to do the period extension in
    :math:`\theta` in the spatial domain.  This is not possible for the MW sampling
    scheme.

    JAX implementation of :func:`~s2fft.resampling.periodic_extension_spatial_mwss`.

    Args:
        f (jnp.ndarray): Signal on the sphere sampled with MWSS sampling scheme.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        jnp.ndarray: Signal on the sphere extended to :math:`\theta` domain
        :math:`[0,2\pi]`, in MWSS sampling scheme.

    """
    ntheta = L + 1
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = jnp.zeros((f.shape[0], ntheta_ext, nphi), dtype=compatible_complex_dtype(f))
    f_ext = f_ext.at[:, 0:ntheta, 0:nphi].set(f[:, 0:ntheta, 0:nphi])
    if spin.size > 1:
        f_ext = f_ext.at[:, ntheta:, 0 : 2 * L].set(
            jnp.einsum(
                "btp,b->btp",
                jnp.fft.fftshift(
                    jnp.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], axis=-2), axes=-1
                ),
                (-1) ** spin,
                optimize=True,
            ),
        )
    else:
        f_ext = f_ext.at[:, ntheta:, 0 : 2 * L].set(
            (-1) ** spin
            * jnp.fft.fftshift(
                jnp.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], axis=-2), axes=-1
            ),
        )
    return f_ext
