from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax
from s2fft import recursions
from warnings import warn


def spin_spherical_kernel(
    L: int,
    spin: int = 0,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
):
    r"""Precompute the wigner-d kernel for spin-spherical transform. This can be
    drastically faster but comes at a :math:`\mathcal{O}(L^3)` memory overhead, making
    it infeasible for :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        np.ndarray: Transform kernel for spin-spherical harmonic transform.
    """
    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform."
        )
    m_start_ind = L - 1 if reality else 0
    m_dim = L if reality else 2 * L - 1

    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    dl = np.zeros((len(thetas), L, m_dim), dtype=np.float64)
    for t, theta in enumerate(thetas):
        for el in range(abs(spin), L):
            dl[t, el] = recursions.turok.compute_slice(theta, el, L, -spin, reality)[
                m_start_ind:
            ]
            dl[t, el] *= np.sqrt((2 * el + 1) / (4 * np.pi))

    if forward:
        weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
        dl = np.einsum("...tlm, ...t->...tlm", dl, weights)

    if sampling.lower() == "healpix":
        dl = np.einsum(
            "...tlm,...tm->...tlm",
            dl,
            healpix_phase_shifts(L, nside, forward)[:, m_start_ind:],
        )

    return dl


def spin_spherical_kernel_jax(
    L: int,
    spin: int = 0,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
):
    r"""Precompute the wigner-d kernel for spin-spherical transform. This can be
    drastically faster but comes at a :math:`\mathcal{O}(L^3)` memory overhead, making
    it infeasible for :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        jnp.ndarray: Transform kernel for spin-spherical harmonic transform.
    """
    m_start_ind = L - 1 if reality else 0

    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    dl = recursions.price_mcewen.compute_all_slices_jax(
        thetas, L, spin, sampling, forward, nside
    )
    dl = dl.at[jnp.where(dl != dl)].set(0)
    dl = jnp.swapaxes(dl, 0, 2)
    dl = jnp.swapaxes(dl, 0, 1)

    # North pole singularity
    if sampling.lower() == "mwss":
        dl = dl.at[0].set(0)
        dl = dl = dl.at[0, :, L - 1 - spin].set(1)

    # South pole singularity
    if sampling.lower() in ["mw", "mwss"]:
        dl = dl.at[-1].set(0)
        dl = dl.at[-1, :, L - 1 + spin].set((-1) ** (jnp.arange(L) - spin))

    dl = dl[:, :, m_start_ind:]

    scaling = jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi))
    dl = jnp.einsum("...tlm,...l->...tlm", dl, scaling, optimize=True)

    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
        dl = jnp.einsum("...tlm, ...t->...tlm", dl, weights, optimize=True)

    if sampling.lower() == "healpix":
        dl = jnp.einsum(
            "...tlm,...tm->...tlm",
            dl,
            healpix_phase_shifts(L, nside, forward)[:, m_start_ind:],
            optimize=True,
        )

    return dl


def wigner_kernel(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
):
    r"""Precompute the wigner-d kernels required for a Wigner transform. This can be
    drastically faster but comes at a :math:`\mathcal{O}(NL^3)` memory overhead, making
    it infeasible for :math:`L \geq 512`.

    Args:
        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        np.ndarray: Transform kernel for Wigner transform.
    """
    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    dl = np.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=np.float64)
    for n in range(n_start_ind - N + 1, N):
        for t, theta in enumerate(thetas):
            for el in range(abs(n), L):
                ind = n if reality else N - 1 + n
                dl[ind, t, el] = recursions.turok.compute_slice(theta, el, L, n, False)

    if forward:
        weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
        dl = np.einsum("...ntlm, ...t->...ntlm", dl, weights)
        dl *= 2 * np.pi / (2 * N - 1)

    else:
        dl = np.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * np.arange(L) + 1) / (8 * np.pi**2),
        )

    if sampling.lower() == "healpix":
        dl = np.einsum(
            "...ntlm,...tm->...ntlm",
            dl,
            healpix_phase_shifts(L, nside, forward),
        )

    return dl


def wigner_kernel_jax(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
):
    r"""Precompute the wigner-d kernels required for a Wigner transform. This can be
    drastically faster but comes at a :math:`\mathcal{O}(NL^3)` memory overhead, making
    it infeasible for :math:`L \geq 512`.

    Args:
        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        jnp.ndarray: Transform kernel for Wigner transform.
    """
    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    dl = jnp.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=np.float64)
    for n in range(n_start_ind - N + 1, N):
        ind = n if reality else N - 1 + n
        dl_n = recursions.price_mcewen.compute_all_slices_jax(
            thetas, L, -n, sampling, forward, nside
        )
        dl_n = dl_n.at[jnp.where(dl_n != dl_n)].set(0)
        dl_n = jnp.swapaxes(dl_n, 0, 2)
        dl_n = jnp.swapaxes(dl_n, 0, 1)

        # North pole singularity
        if sampling.lower() == "mwss":
            dl_n = dl_n.at[0].set(0)
            dl_n = dl_n = dl_n.at[0, :, L - 1 + n].set(1)

        # South pole singularity
        if sampling.lower() in ["mw", "mwss"]:
            dl_n = dl_n.at[-1].set(0)
            dl_n = dl_n.at[-1, :, L - 1 - n].set((-1) ** (jnp.arange(L) + n))

        # Remove l <= n
        dl_n = dl_n.at[:, : abs(n), :].set(0)
        dl = dl.at[ind].add(dl_n)

    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
        dl = jnp.einsum("...ntlm, ...t->...ntlm", dl, weights, optimize=True)
        dl *= 2 * jnp.pi / (2 * N - 1)

    else:
        dl = jnp.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * jnp.arange(L) + 1) / (8 * jnp.pi**2),
            optimize=True,
        )

    if sampling.lower() == "healpix":
        dl = np.einsum(
            "...ntlm,...tm->...ntlm",
            dl,
            healpix_phase_shifts(L, nside, forward),
        )

    return dl


def healpix_phase_shifts(
    L: int,
    nside: int,
    forward: bool = False,
) -> np.ndarray:
    r"""Generates a phase shift vector for HEALPix for all :math:`\theta` rings.

    Args:
        L (int, optional): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        np.ndarray: Vector of phase shifts with shape :math:`[thetas, 2L-1]`.
    """
    thetas = samples.thetas(L, "healpix", nside)
    phase_array = np.zeros((len(thetas), 2 * L - 1), dtype=np.complex128)
    for t, theta in enumerate(thetas):
        phase_array[t] = samples.ring_phase_shift_hp(L, t, nside, forward)

    return phase_array
