from jax import jit, custom_vjp

import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import List
from s2fft.sampling import s2_samples as samples
from s2fft.utils import (
    resampling,
    quadrature,
    resampling_jax,
    quadrature_jax,
)
from s2fft.utils import healpix_ffts as hp
from s2fft.transforms import otf_recursions as otf


def inverse(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Wrapper for the inverse spin-spherical harmonic transform.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Raises:
        ValueError: Transform method not recognised.

    Returns:
        np.ndarray: Signal on the sphere.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    if method == "numpy":
        return inverse_numpy(flm, L, spin, nside, sampling, reality, precomps, L_lower)
    elif method == "jax":
        return inverse_jax(
            flm, L, spin, nside, sampling, reality, precomps, spmd, L_lower
        )
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


def inverse_numpy(
    flm: np.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Compute the inverse spin-spherical harmonic transform (numpy).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(L^2)`.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Signal on the sphere.
    """
    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling, nside)
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0
    m_start_ind = L - 1 if reality else 0
    L0 = L_lower

    # Apply harmonic normalisation
    flm[L0:] = np.einsum(
        "lm,l->lm", flm[L0:], np.sqrt((2 * np.arange(L0, L) + 1) / (4 * np.pi))
    )

    # Perform latitudinal wigner-d recursions
    ftm = otf.inverse_latitudinal_step(
        flm, thetas, L, spin, nside, sampling, reality, precomps, L0
    )

    # Remove south pole singularity
    if sampling.lower() in ["mw", "mwss"]:
        ftm[-1] = 0
        ftm[-1, L - 1 + spin + m_offset] = np.nansum(
            (-1) ** abs(np.arange(L0, L) - spin) * flm[L0:, L - 1 + spin]
        )
    # Remove north pole singularity
    if sampling.lower() == "mwss":
        ftm[0] = 0
        ftm[0, L - 1 - spin + m_offset] = jnp.nansum(flm[L0:, L - 1 - spin])

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp(L, nside, False, reality)
        ftm[:, m_start_ind + m_offset :] *= phase_shifts

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    if sampling.lower() == "healpix":
        if reality:
            ftm[:, m_offset : L - 1 + m_offset] = np.flip(
                np.conj(ftm[:, L - 1 + m_offset + 1 :]), axis=-1
            )
        return hp.healpix_ifft(ftm, L, nside, "numpy", reality)
    else:
        if reality:
            return np.fft.irfft(
                ftm[:, L - 1 + m_offset :],
                samples.nphi_equiang(L, sampling),
                axis=1,
                norm="forward",
            )
        else:
            return np.fft.ifft(np.fft.ifftshift(ftm, axes=1), axis=1, norm="forward")


@partial(jit, static_argnums=(1, 3, 4, 5, 7, 8))
def inverse_jax(
    flm: jnp.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""Compute the inverse spin-spherical harmonic transform (JAX).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(L^2)`. This is a JAX implementation of
    :func:`~inverse_numpy`.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices. Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Signal on the sphere.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    # Define latitudinal sample positions and Fourier offsets
    thetas = samples.thetas(L, sampling, nside)
    m_offset = 1 if sampling.lower() in ["mwss", "healpix"] else 0
    m_start_ind = L - 1 if reality else 0

    # Apply harmonic normalisation
    flm = flm.at[L_lower:].set(
        jnp.einsum(
            "lm,l->lm",
            flm[L_lower:],
            jnp.sqrt((2 * jnp.arange(L_lower, L) + 1) / (4 * jnp.pi)),
            optimize=True,
        )
    )

    # Perform latitudinal wigner-d recursions
    @custom_vjp
    def flm_to_ftm(flm, spin, precomps):
        return otf.inverse_latitudinal_step_jax(
            flm,
            thetas,
            L,
            spin,
            nside,
            sampling,
            reality,
            precomps=precomps,
            spmd=spmd,
            L_lower=L_lower,
        )

    def f_fwd(flm, spin, precomps):
        return flm_to_ftm(flm, spin, precomps), ([], spin, [])

    def f_bwd(res, gtm):
        spin = res[1]
        glm = otf.forward_latitudinal_step_jax(
            gtm,
            thetas,
            L,
            spin,
            nside,
            sampling,
            reality,
            spmd=spmd,
            L_lower=L_lower,
        )
        return glm, None, None

    flm_to_ftm.defvjp(f_fwd, f_bwd)
    ftm = flm_to_ftm(flm, spin, precomps)

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp_jax(L, nside, False, reality)
        ftm = ftm.at[:, m_start_ind + m_offset :].multiply(phase_shifts)

    # Perform longitundal Fast Fourier Transforms
    ftm *= (-1) ** spin
    if reality:
        ftm = ftm.at[:, m_offset : L - 1 + m_offset].set(
            jnp.flip(jnp.conj(ftm[:, L - 1 + m_offset + 1 :]), axis=-1)
        )
    if sampling.lower() == "healpix":
        return hp.healpix_ifft(ftm, L, nside, "jax")
    else:
        ftm = jnp.conj(jnp.fft.ifftshift(ftm, axes=1))
        f = jnp.conj(jnp.fft.fft(ftm, axis=1, norm="backward"))
        return jnp.real(f) if reality else f


def forward(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Wrapper for the forward spin-spherical harmonic transform.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Raises:
        ValueError: Transform method not recognised.

    Returns:
        np.ndarray: Spherical harmonic coefficients.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    if method == "numpy":
        return forward_numpy(f, L, spin, nside, sampling, reality, precomps, L_lower)
    elif method == "jax":
        return forward_jax(
            f, L, spin, nside, sampling, reality, precomps, spmd, L_lower
        )
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


def forward_numpy(
    f: np.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Compute the forward spin-spherical harmonic transform (JAX).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(L^2)`.

    Args:
        f (np.ndarray): Signal on the sphere

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        np.ndarray: Spherical harmonic coefficients
    """
    # Resample mw onto mwss and double resolution of both
    if sampling.lower() == "mw":
        f = resampling.mw_to_mwss(f, L, spin)
    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Define latitudinal sample positions and Fourier offsets
    weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    m_start_ind = L - 1 if reality else 0
    L0 = L_lower

    # Perform longitundal Fast Fourier Transforms
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside, "numpy", reality)
    else:
        if reality:
            t = np.fft.rfft(np.real(f), axis=1, norm="backward")
            if m_offset != 0:
                t = t[:, :-1]
            ftm = np.zeros_like(f).astype(np.complex128)
            ftm[:, L - 1 + m_offset :] = t
        else:
            ftm = np.fft.fftshift(np.fft.fft(f, axis=1, norm="backward"), axes=1)

    # Apply quadrature weights
    ftm = np.einsum("tm,t->tm", ftm, weights)

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp(L, nside, True, reality)
        ftm[:, m_start_ind + m_offset :] *= phase_shifts

    # Perform latitudinal wigner-d recursions
    if sampling.lower() == "mwss":
        flm = otf.forward_latitudinal_step(
            ftm[1:-1],
            thetas[1:-1],
            L,
            spin,
            nside,
            sampling,
            reality,
            precomps,
            L0,
        )
    else:
        flm = otf.forward_latitudinal_step(
            ftm, thetas, L, spin, nside, sampling, reality, precomps, L0
        )

    # Include both pole singularities explicitly
    if sampling.lower() == "mwss":
        flm[L0:, L - 1 + spin] += (-1) ** abs(np.arange(L0, L) - spin) * ftm[
            -1, L - 1 + spin + m_offset
        ]
        flm[L0:, L - 1 - spin] += ftm[0, L - 1 - spin + m_offset]

    # Apply harmonic normalisation
    flm[L0:] = np.einsum(
        "lm,l->lm", flm[L0:], np.sqrt((2 * np.arange(L0, L) + 1) / (4 * np.pi))
    )

    # Mirror to complete hermitian conjugate
    if reality:
        m_conj = (-1) ** (np.arange(1, L) % 2)
        flm[..., :m_start_ind] = np.flip(
            m_conj * np.conj(flm[..., m_start_ind + 1 :]), axis=-1
        )

    # Enforce spin condition explicitly
    flm[: max(abs(spin), L_lower)] = 0.0

    return flm * (-1) ** spin


@partial(jit, static_argnums=(1, 3, 4, 5, 7, 8))
def forward_jax(
    f: jnp.ndarray,
    L: int,
    spin: int = 0,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""Compute the forward spin-spherical harmonic transform (JAX).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(L^2)`. This is a JAX implementation of
    :func:`~forward_numpy`.

    Args:
        f (jnp.ndarray): Signal on the sphere

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices. Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Spherical harmonic coefficients

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    # Resample mw onto mwss and double resolution of both
    if sampling.lower() == "mw":
        f = resampling_jax.mw_to_mwss(f, L, spin)
    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        f = resampling_jax.upsample_by_two_mwss(f, L, spin)
        thetas = samples.thetas(2 * L, sampling)
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Define latitudinal sample positions and Fourier offsets
    weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    m_start_ind = L - 1 if reality else 0

    # Perform longitundal Fast Fourier Transforms
    if sampling.lower() == "healpix":
        ftm = hp.healpix_fft(f, L, nside, "jax", reality)
    else:
        if reality:
            t = jnp.fft.rfft(jnp.real(f), axis=1, norm="backward")
            if m_offset != 0:
                t = t[:, :-1]
            ftm = jnp.zeros_like(f).astype(jnp.complex128)
            ftm = ftm.at[:, L - 1 + m_offset :].set(t)
        else:
            ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1)

    # Apply quadrature weights
    ftm = jnp.einsum("tm,t->tm", ftm, weights, optimize=True)

    # Correct healpix theta row offsets
    if sampling.lower() == "healpix":
        phase_shifts = hp.ring_phase_shifts_hp_jax(L, nside, True, reality)
        ftm = ftm.at[:, m_start_ind + m_offset :].multiply(phase_shifts)

    # Perform latitudinal wigner-d recursions
    @custom_vjp
    def ftm_to_flm(ftm, spin, precomps):
        flm = otf.forward_latitudinal_step_jax(
            ftm,
            thetas,
            L,
            spin,
            nside,
            sampling,
            reality,
            precomps=precomps,
            spmd=spmd,
            L_lower=L_lower,
        )
        return flm

    def f_fwd(ftm, spin, precomps):
        return ftm_to_flm(ftm, spin, precomps), ([], spin, [])

    def f_bwd(res, glm):
        spin = res[1]
        gtm = otf.inverse_latitudinal_step_jax(
            glm,
            thetas,
            L,
            spin,
            nside,
            sampling,
            reality,
            spmd=spmd,
            L_lower=L_lower,
        )
        return gtm, None, None

    ftm_to_flm.defvjp(f_fwd, f_bwd)
    flm = ftm_to_flm(ftm, spin, precomps)

    # Apply harmonic normalisation
    flm = flm.at[L_lower:].set(
        jnp.einsum(
            "lm,l->lm",
            flm[L_lower:],
            jnp.sqrt((2 * jnp.arange(L_lower, L) + 1) / (4 * jnp.pi)),
            optimize=True,
        )
    )

    # Hermitian conjugate symmetry
    if reality:
        flm = flm.at[..., :m_start_ind].set(
            jnp.flip(
                (-1) ** (jnp.arange(1, L) % 2) * jnp.conj(flm[..., m_start_ind + 1 :]),
                axis=-1,
            )
        )

    # Enforce spin condition explicitly.
    indices = jnp.repeat(jnp.expand_dims(jnp.arange(L), -1), 2 * L - 1, axis=-1)
    flm = jnp.where(indices < abs(spin), jnp.zeros_like(flm), flm[..., :])

    return flm * (-1) ** spin
