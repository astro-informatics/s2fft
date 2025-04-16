from functools import partial
from typing import List, Optional

import jax.numpy as jnp
import numpy as np
from jax import custom_vjp, jit

from s2fft.sampling import s2_samples as samples
from s2fft.transforms import c_backend_spherical as c_sph
from s2fft.transforms import otf_recursions as otf
from s2fft.utils import healpix_ffts as hp
from s2fft.utils import (
    iterative_refinement,
    quadrature,
    quadrature_jax,
    resampling,
    resampling_jax,
)


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
    _ssht_backend: int = 1,
) -> np.ndarray:
    r"""
    Wrapper for the inverse spin-spherical harmonic transform.

    Args:
        flm (np.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax", "jax_ssht", "jax_healpy"}.
            Defaults to "numpy".

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

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

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
    if method not in _inverse_functions:
        raise ValueError(f"Method {method} not recognised.")

    if spin >= 8 and method in ["numpy", "jax"]:
        raise Warning("Recursive transform may provide lower precision beyond spin ~ 8")

    inverse_kwargs = {"flm": flm, "L": L}
    if method in ("numpy", "jax"):
        inverse_kwargs.update(sampling=sampling, precomps=precomps, L_lower=L_lower)
    if method == "jax":
        inverse_kwargs["spmd"] = spmd
    if method == "jax_healpy":
        if sampling.lower() != "healpix":
            raise ValueError("Healpy only supports healpix sampling.")
    else:
        inverse_kwargs.update(spin=spin, reality=reality)
    if method == "jax_ssht":
        if sampling.lower() == "healpix":
            raise ValueError("SSHT does not support healpix sampling.")
        ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())
        inverse_kwargs.update(ssht_sampling=ssht_sampling, _ssht_backend=_ssht_backend)
    else:
        inverse_kwargs["nside"] = nside

    return _inverse_functions[method](**inverse_kwargs)


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
    r"""
    Compute the inverse spin-spherical harmonic transform (numpy).

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
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

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

    # Copy flm argument to avoid in-place updates being propagated back to caller
    flm = flm.copy()

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
    r"""
    Compute the inverse spin-spherical harmonic transform (JAX).

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
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

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
    ftm *= (-1) ** jnp.abs(spin)
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
    nside: Optional[int] = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: Optional[List] = None,
    spmd: bool = False,
    L_lower: int = 0,
    iter: Optional[int] = None,
    _ssht_backend: int = 1,
) -> np.ndarray:
    r"""
    Wrapper for the forward spin-spherical harmonic transform.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax", "jax_ssht", "jax_healpy"}.
            Defaults to "numpy".

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

        iter (int, optional): Number of iterative refinement iterations to use to
            improve accuracy of forward transform (as an inverse of inverse transform).
            Primarily of use with HEALPix sampling for which there is not a sampling
            theorem, and round-tripping through the forward and inverse transforms will
            introduce an error. If set to `None`, the default, 3 iterations will be used
            if :code:`sampling == "healpix"` and :code:`method == "jax_healpy"` and zero
            otherwise. Not used for `jax_ssht` method.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

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
    if spin >= 8 and method in ["numpy", "jax"]:
        raise Warning("Recursive transform may provide lower precision beyond spin ~ 8")

    if iter is None:
        iter = 3 if sampling.lower() == "healpix" and method == "jax_healpy" else 0
    if method in {"numpy", "jax", "cuda"}:
        common_kwargs = {
            "L": L,
            "spin": spin,
            "nside": nside,
            "sampling": sampling,
            "reality": reality,
            "L_lower": L_lower,
        }
        forward_kwargs = {**common_kwargs, "precomps": precomps}
        inverse_kwargs = common_kwargs
        if method in {"jax", "cuda"}:
            forward_kwargs["spmd"] = spmd
            forward_kwargs["use_healpix_custom_primitive"] = method == "cuda"
            inverse_kwargs["method"] = "jax"
            inverse_kwargs["spmd"] = spmd
            forward_function = forward_jax
        else:
            inverse_kwargs["method"] = "numpy"
            forward_function = forward_numpy
        return iterative_refinement.forward_with_iterative_refinement(
            f=f,
            n_iter=iter,
            forward_function=partial(forward_function, **forward_kwargs),
            backward_function=partial(inverse, **inverse_kwargs),
        )
    elif method == "jax_ssht":
        if sampling.lower() == "healpix":
            raise ValueError("SSHT does not support healpix sampling.")
        ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())
        return c_sph.ssht_forward(f, L, spin, reality, ssht_sampling, _ssht_backend)
    elif method == "jax_healpy":
        if sampling.lower() != "healpix":
            raise ValueError("Healpy only supports healpix sampling.")
        return c_sph.healpy_forward(f, L, nside, iter)
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
    r"""
    Compute the forward spin-spherical harmonic transform (JAX).

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
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

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


@partial(jit, static_argnums=(1, 3, 4, 5, 7, 8, 9))
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
    use_healpix_custom_primitive: bool = False,
) -> jnp.ndarray:
    r"""
    Compute the forward spin-spherical harmonic transform (JAX).

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
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices. Defaults to False.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        use_healpix_custom_primitive (bool, optional): Whether to use a custom CUDA
            primitive for computing HEALPix fast fourier transform when `sampling =
            "healpix"` and running on a cuda compatible gpu device. using a custom
            primitive reduces long compilation times when jit compiling. defaults to
            `False`.

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
        if use_healpix_custom_primitive:
            ftm = hp.healpix_fft(f, L, nside, "cuda", reality)
        else:
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

    return flm * (-1) ** jnp.abs(spin)


_inverse_functions = {
    "numpy": inverse_numpy,
    "jax": inverse_jax,
    "jax_ssht": c_sph.ssht_inverse,
    "jax_healpy": c_sph.healpy_inverse,
}

_forward_functions = {
    "numpy": forward_numpy,
    "jax": forward_jax,
    "jax_ssht": c_sph.ssht_forward,
    "jax_healpy": c_sph.healpy_forward,
}
