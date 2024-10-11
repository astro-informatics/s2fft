from warnings import warn

import jax
import jax.numpy as jnp
import numpy as np
import torch

from s2fft import recursions
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax

# Maximum spin number at which Price-McEwen recursion is sufficiently accurate.
# For spins > PM_MAX_STABLE_SPIN one should default to the Risbo recursion.
PM_MAX_STABLE_SPIN = 6


def spin_spherical_kernel(
    L: int,
    spin: int = 0,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = True,
    using_torch: bool = False,
    recursion: str = "auto",
) -> np.ndarray:
    r"""
    Precompute the wigner-d kernel for spin-spherical transform.

    This implementation is typically faster than computing these elements on-the-fly but
    comes at a :math:`\mathcal{O}(L^3)` memory overhead, making it infeasible for large
    bandlimits :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        using_torch (bool, optional): Desired frontend functionality. Defaults to False.

        recursion (str, optional): Recursion to adopt. Supported recursion schemes include
            {"auto", "price-mcewen", "risbo"}. Defaults to "auto" which will detect the
            most appropriate recursion given the parameter configuration.

    Returns:
        np.ndarray: Transform kernel for spin-spherical harmonic transform.

    """
    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform.",
            stacklevel=2,
        )
    if recursion.lower() == "price-mcewen" and abs(spin) > PM_MAX_STABLE_SPIN:
        raise ValueError(
            f"The Price-McEwen recursion can become unstable for spins >= {PM_MAX_STABLE_SPIN}."
        )

    if recursion.lower() == "auto":
        # This mode automatically determines which recursion is best suited for the
        # current parameter configuration.
        recursion = "risbo" if abs(spin) > PM_MAX_STABLE_SPIN else "price-mcewen"

    dl = []
    m_start_ind = L - 1 if reality else 0
    m_dim = L if reality else 2 * L - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Calculate Wigner d-function elements through the Price-McEwen recursion.
    # - The complexity of this approach is O(L^3).
    # - This approach becomes inaccurate for abs(spins) >= 5.
    if recursion.lower() == "price-mcewen":
        dl = np.zeros((len(thetas), L, m_dim), dtype=np.float64)
        for t, theta in enumerate(thetas):
            for el in range(abs(spin), L):
                dl[t, el] = recursions.turok.compute_slice(
                    theta, el, L, -spin, reality
                )[m_start_ind:]
                dl[t, el] *= np.sqrt((2 * el + 1) / (4 * np.pi))

    # Calculate Wigner d-function elements through the Risbo recursion.
    elif recursion.lower() == "risbo":
        dl = np.zeros((len(thetas), L, m_dim), dtype=np.float64)

        # GL and HP sampling ARE NOT uniform in theta therefore CANNOT be calculated
        # using the Fourier decomposition of Wigner d-functions. Instead they must be
        # manually bootstrapped from the recursion.
        # - The complexity of this approach is O(L^4).
        # - This approach is stable for arbitrary abs(spins) <= L.
        if sampling.lower() in ["healpix", "gl"]:
            delta = np.zeros((len(thetas), 2 * L - 1, 2 * L - 1), dtype=np.float64)
            for el in range(L):
                delta = recursions.risbo.compute_full_vectorised(delta, thetas, L, el)
                dl[:, el] = delta[:, m_start_ind:, L - 1 - spin]

        # MW, MWSS, and DH sampling ARE uniform in theta therefore CAN be calculated
        # using the Fourier decomposition of Wigner d-functions.
        # - The complexity of this approach is O(L^3LogL).
        # - This approach is stable for arbitrary abs(spins) <= L.
        if sampling.lower() in ["mw", "mwss", "dh"]:
            # Number of samples for inverse FFT over Wigner Fourier coefficients.
            if sampling.lower() == "mw":
                nsamps = 2 * len(thetas) - 1
            elif sampling.lower() == "mwss":
                nsamps = 2 * len(thetas) - 2
            elif sampling.lower() == "dh":
                nsamps = 2 * len(thetas)
            delta = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

            # Calculate the Fourier coefficients of the Wigner d-functions, delta(pi/2).
            m_value = np.arange(-L + 1, L)
            for el in range(L):
                delta = recursions.risbo.compute_full(delta, np.pi / 2, L, el)
                temp = np.einsum(
                    "am,a,m->am",
                    delta[:, m_start_ind:],
                    delta[:, L - 1 - spin],
                    1j ** (-spin - m_value[m_start_ind:]),
                )
                temp = np.einsum("am,a->am", temp, np.exp(1j * m_value * thetas[0]))
                temp = np.fft.irfft(temp[L - 1 :], n=nsamps, axis=0, norm="forward")

                dl[:, el] = temp[: len(thetas)]

        # Fold in normalisation to avoid recomputation at run-time.
        dl = np.einsum("tlm,l->tlm", dl, np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi)))

    else:
        raise ValueError(f"Recursion method {recursion} not recognised.")

    # Fold in quadrature to avoid recomputation at run-time.
    if forward:
        weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
        dl = np.einsum("...tlm, ...t->...tlm", dl, weights)

    # Apply the per ring phase shift for healpix sampling.
    if sampling.lower() == "healpix":
        dl = np.einsum(
            "...tlm,...tm->...tlm",
            dl,
            healpix_phase_shifts(L, nside, forward)[:, m_start_ind:],
        )

    return torch.from_numpy(dl) if using_torch else dl


def spin_spherical_kernel_jax(
    L: int,
    spin: int = 0,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = True,
    recursion: str = "auto",
) -> jnp.ndarray:
    r"""
    Precompute the wigner-d kernel for spin-spherical transform.

    This implementation is typically faster than computing these elements on-the-fly but
    comes at a :math:`\mathcal{O}(L^3)` memory overhead, making it infeasible for large
    bandlimits :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        spin (int): Harmonic spin.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        recursion (str, optional): Recursion to adopt. Supported recursion schemes include
            {"auto", "price-mcewen", "risbo"}. Defaults to "auto" which will detect the
            most appropriate recursion given the parameter configuration.

    Returns:
        jnp.ndarray: Transform kernel for spin-spherical harmonic transform.

    """
    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defaulting to complex transform.",
            stacklevel=2,
        )
    if recursion.lower() == "price-mcewen" and abs(spin) > PM_MAX_STABLE_SPIN:
        raise ValueError(
            f"The Price-McEwen recursion can become unstable for spins >= {PM_MAX_STABLE_SPIN}."
        )

    if recursion.lower() == "auto":
        # This mode automatically determines which recursion is best suited for the
        # current parameter configuration.
        recursion = "risbo" if abs(spin) > PM_MAX_STABLE_SPIN else "price-mcewen"

    dl = []
    m_start_ind = L - 1 if reality else 0
    m_dim = L if reality else 2 * L - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Calculate Wigner d-function elements through the Price-McEwen recursion.
    # - The complexity of this approach is O(L^3).
    # - This approach becomes inaccurate for abs(spins) >= 5.
    if recursion.lower() == "price-mcewen":
        dl = recursions.price_mcewen.compute_all_slices_jax(
            thetas, L, spin, sampling, forward, nside
        )
        dl = dl.at[jnp.where(dl != dl)].set(0)
        dl = jnp.swapaxes(dl, 0, 2)
        dl = jnp.swapaxes(dl, 0, 1)

        # North pole singularity
        if sampling.lower() == "mwss":
            dl = dl.at[0].set(0)
            dl = dl.at[0, :, L - 1 - spin].set(1)

        # South pole singularity
        if sampling.lower() in ["mw", "mwss"]:
            dl = dl.at[-1].set(0)
            dl = dl.at[-1, :, L - 1 + spin].set((-1) ** (jnp.arange(L) - spin))
        dl = dl.at[:, : jnp.abs(spin)].multiply(0)

        dl = dl[:, :, m_start_ind:]

    # Calculate Wigner d-function elements through the Risbo recursion.
    elif recursion.lower() == "risbo":
        dl = jnp.zeros((len(thetas), L, m_dim), dtype=jnp.float64)

        # GL and HP sampling ARE NOT uniform in theta therefore CANNOT be calculated
        # using the Fourier decomposition of Wigner d-functions. Instead they must be
        # manually bootstrapped from the recursion.
        # - The complexity of this approach is O(L^4).
        # - This approach is stable for arbitrary abs(spins) <= L.
        if sampling.lower() in ["healpix", "gl"]:
            delta = jnp.zeros((len(thetas), 2 * L - 1, 2 * L - 1), dtype=jnp.float64)
            vfunc = jax.vmap(
                recursions.risbo_jax.compute_full, in_axes=(0, 0, None, None)
            )
            for el in range(L):
                delta = vfunc(delta, thetas, L, el)
                dl = dl.at[:, el].set(delta[:, m_start_ind:, L - 1 - spin])

        # MW, MWSS, and DH sampling ARE uniform in theta therefore CAN be calculated
        # using the Fourier decomposition of Wigner d-functions.
        # - The complexity of this approach is O(L^3LogL).
        # - This approach is stable for arbitrary abs(spins) <= L.
        elif sampling.lower() in ["mw", "mwss", "dh"]:
            # Number of samples for inverse FFT over Wigner Fourier coefficients.
            if sampling.lower() == "mw":
                nsamps = 2 * len(thetas) - 1
            elif sampling.lower() == "mwss":
                nsamps = 2 * len(thetas) - 2
            elif sampling.lower() == "dh":
                nsamps = 2 * len(thetas)
            delta = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)

            # Calculate the Fourier coefficients of the Wigner d-functions, delta(pi/2).
            for el in range(L):
                delta = recursions.risbo_jax.compute_full(delta, jnp.pi / 2, L, el)
                m_value = jnp.arange(-L + 1, L)
                temp = jnp.einsum(
                    "am,a,m->am",
                    delta[:, m_start_ind:],
                    delta[:, L - 1 - spin],
                    1j ** (-spin - m_value[m_start_ind:]),
                )
                temp = jnp.einsum("am,a->am", temp, jnp.exp(1j * m_value * thetas[0]))
                temp = jnp.fft.irfft(temp[L - 1 :], n=nsamps, axis=0, norm="forward")

                dl = dl.at[:, el].set(temp[: len(thetas)])

    else:
        raise ValueError(f"Recursion method {recursion} not recognised.")

    # Fold in normalisation to avoid recomputation at run-time.
    dl = jnp.einsum("tlm,l->tlm", dl, jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)))

    # Fold in quadrature to avoid recomputation at run-time.
    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
        dl = jnp.einsum("...tlm, ...t->...tlm", dl, weights)

    # Apply the per ring phase shift for healpix sampling.
    if sampling.lower() == "healpix":
        dl = jnp.einsum(
            "...tlm,...tm->...tlm",
            dl,
            healpix_phase_shifts(L, nside, forward)[:, m_start_ind:],
        )

    return dl


def wigner_kernel(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    mode: str = "auto",
    using_torch: bool = False,
) -> np.ndarray:
    r"""
    Precompute the wigner-d kernel for Wigner transform.

    This implementation is typically faster than computing these elements on-the-fly but
    comes at a :math:`\mathcal{O}(NL^3)` memory overhead, making it infeasible for large
    bandlimits :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        mode (str, optional): Whether to use FFT approach or manually compute each element.
            {"auto", "direct", "fft"}. Defaults to "auto" which will detect the
            most appropriate recursion given the parameter configuration.

        using_torch (bool, optional): Desired frontend functionality. Defaults to False.

    Returns:
        np.ndarray: Transform kernel for Wigner transform.

    """
    if mode.lower() == "fft" and sampling.lower() not in ["mw", "mwss", "dh"]:
        raise ValueError(
            f"Fourier based recursion is not valid for {sampling} sampling."
        )
    # Determine operational mode automatically.
    # - Can only use the FFT approach when uniformly sampling in theta.
    # - FFT approach is only more efficient when N <= L/Log(L) roughly.
    if mode.lower() == "auto":
        if sampling.lower() in ["mw", "mwss", "dh"]:
            mode = "fft" if N <= int(L / np.log(L)) else "direct"
        else:
            mode = "direct"

    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Range values which need only be defined once.
    m_value = np.arange(-L + 1, L)
    n = np.arange(n_start_ind - N + 1, N)
    dl = np.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=np.float64)

    # GL and HP sampling ARE NOT uniform in theta therefore CANNOT be calculated
    # using the Fourier decomposition of Wigner d-functions. Instead they must be
    # manually calculated from the recursion.
    # - The complexity of this approach is ALWAYS O(L^4).
    # - This approach is stable for arbitrary abs(spins) <= L.
    if mode.lower() == "direct":
        delta = np.zeros((len(thetas), 2 * L - 1, 2 * L - 1), dtype=np.float64)
        for el in range(L):
            delta = recursions.risbo.compute_full_vectorised(delta, thetas, L, el)
            dl[:, :, el] = np.moveaxis(delta, -1, 0)[L - 1 + n]

    # MW, MWSS, and DH sampling ARE uniform in theta therefore CAN be calculated
    # using the Fourier decomposition of Wigner d-functions.
    # - The complexity of this approach is O(NL^3LogL).
    # - This approach is stable for arbitrary abs(spins) <= L.
    # Therefore when NL^3LogL <= L^4 i.e. when N <= L/LogL, the Fourier based approach
    # is more efficient. This can be a large difference for large L >> N.
    elif mode.lower() == "fft":
        # Number of samples for inverse FFT over Wigner Fourier coefficients.
        if sampling.lower() == "mw":
            nsamps = 2 * len(thetas) - 1
        elif sampling.lower() == "mwss":
            nsamps = 2 * len(thetas) - 2
        elif sampling.lower() == "dh":
            nsamps = 2 * len(thetas)
        delta = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

        # Calculate the Fourier coefficients of the Wigner d-functions, delta(pi/2).
        for el in range(L):
            delta = recursions.risbo.compute_full(delta, np.pi / 2, L, el)
            temp = np.einsum(
                "am,an,m,n->amn",
                delta,
                delta[:, L - 1 + n],
                1j ** (-m_value),
                1j ** (n),
            )
            temp = np.einsum("amn,a->amn", temp, np.exp(1j * m_value * thetas[0]))
            temp = np.fft.irfft(temp[L - 1 :], n=nsamps, axis=0, norm="forward")
            dl[:, :, el] = np.moveaxis(temp[: len(thetas)], -1, 0)

    else:
        raise ValueError(f"Recursion method {mode} not recognised.")

    # Fold in quadrature to avoid recomputation at run-time (forward).
    if forward:
        weights = quadrature.quad_weights_transform(L, sampling, 0, nside)
        dl = np.einsum("...ntlm, ...t->...ntlm", dl, weights)
        dl *= 2 * np.pi / (2 * N - 1)

    # Fold in normalisation to avoid recomputation at run-time (inverse).
    else:
        dl = np.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * np.arange(L) + 1) / (8 * np.pi**2),
        )

    # Apply the per ring phase shift for healpix sampling.
    if sampling.lower() == "healpix":
        dl = np.einsum(
            "...ntlm,...tm->...ntlm",
            dl,
            healpix_phase_shifts(L, nside, forward),
        )

    return torch.from_numpy(dl) if using_torch else dl


def wigner_kernel_jax(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    mode: str = "auto",
) -> jnp.ndarray:
    r"""
    Precompute the wigner-d kernel for Wigner transform.

    This implementation is typically faster than computing these elements on-the-fly but
    comes at a :math:`\mathcal{O}(NL^3)` memory overhead, making it infeasible for large
    bandlimits :math:`L\geq 512`.

    Args:
        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}. Defaults to "mw".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        mode (str, optional): Whether to use FFT approach or manually compute each element.
            {"auto", "direct", "fft"}. Defaults to "auto" which will detect the
            most appropriate recursion given the parameter configuration.

    Returns:
        jnp.ndarray: Transform kernel for Wigner transform.

    """
    if mode.lower() == "fft" and sampling.lower() not in ["mw", "mwss", "dh"]:
        raise ValueError(
            f"Fourier based recursion is not valid for {sampling} sampling."
        )
    # Determine operational mode automatically.
    # - Can only use the FFT approach when uniformly sampling in theta.
    # - FFT approach is only more efficient when N <= L/Log(L) roughly.
    if mode.lower() == "auto":
        if sampling.lower() in ["mw", "mwss", "dh"]:
            mode = "fft" if N <= int(L / np.log(L)) else "direct"
        else:
            mode = "direct"

    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling, nside)

    # Range values which need only be defined once.
    m_value = jnp.arange(-L + 1, L)
    n = jnp.arange(n_start_ind - N + 1, N)
    dl = jnp.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=jnp.float64)

    # GL and HP sampling ARE NOT uniform in theta therefore CANNOT be calculated
    # using the Fourier decomposition of Wigner d-functions. Instead they must be
    # manually calculated from the recursion.
    # - The complexity of this approach is ALWAYS O(L^4).
    # - This approach is stable for arbitrary abs(spins) <= L.
    if mode.lower() == "direct":
        delta = jnp.zeros((len(thetas), 2 * L - 1, 2 * L - 1), dtype=jnp.float64)
        vfunc = jax.vmap(recursions.risbo_jax.compute_full, in_axes=(0, 0, None, None))
        for el in range(L):
            delta = vfunc(delta, thetas, L, el)
            dl = dl.at[:, :, el].set(jnp.moveaxis(delta, -1, 0)[L - 1 + n])

    # MW, MWSS, and DH sampling ARE uniform in theta therefore CAN be calculated
    # using the Fourier decomposition of Wigner d-functions.
    # - The complexity of this approach is O(NL^3LogL).
    # - This approach is stable for arbitrary abs(spins) <= L.
    # Therefore when NL^3LogL <= L^4 i.e. when N <= L/LogL, the Fourier based approach
    # is more efficient. This can be a large difference for large L >> N.
    elif mode.lower() == "fft":
        # Number of samples for inverse FFT over Wigner Fourier coefficients.
        if sampling.lower() == "mw":
            nsamps = 2 * len(thetas) - 1
        elif sampling.lower() == "mwss":
            nsamps = 2 * len(thetas) - 2
        elif sampling.lower() == "dh":
            nsamps = 2 * len(thetas)
        delta = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)

        # Calculate the Fourier coefficients of the Wigner d-functions, delta(pi/2).
        for el in range(L):
            delta = recursions.risbo_jax.compute_full(delta, jnp.pi / 2, L, el)
            temp = jnp.einsum(
                "am,an,m,n->amn",
                delta,
                delta[:, L - 1 + n],
                1j ** (-m_value),
                1j ** (n),
            )
            temp = jnp.einsum("amn,a->amn", temp, jnp.exp(1j * m_value * thetas[0]))
            temp = jnp.fft.irfft(temp[L - 1 :], n=nsamps, axis=0, norm="forward")
            dl = dl.at[:, :, el].set(jnp.moveaxis(temp[: len(thetas)], -1, 0))

    else:
        raise ValueError(f"Recursion method {mode} not recognised.")

    # Fold in quadrature to avoid recomputation at run-time (forward).
    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling, nside)
        dl = jnp.einsum("...ntlm, ...t->...ntlm", dl, weights)
        dl *= 2 * jnp.pi / (2 * N - 1)

    # Fold in normalisation to avoid recomputation at run-time (inverse).
    else:
        dl = jnp.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * jnp.arange(L) + 1) / (8 * jnp.pi**2),
        )

    # Apply the per ring phase shift for healpix sampling.
    if sampling.lower() == "healpix":
        dl = jnp.einsum(
            "...ntlm,...tm->...ntlm",
            dl,
            healpix_phase_shifts(L, nside, forward),
        )

    return dl


def healpix_phase_shifts(L: int, nside: int, forward: bool = False) -> np.ndarray:
    r"""
    Generates a phase shift vector for HEALPix for all :math:`\theta` rings.

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
    for t in range(len(thetas)):
        phase_array[t] = samples.ring_phase_shift_hp(L, t, nside, forward)

    return phase_array
