import jax

jax.config.update("jax_enable_x64", True)

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
    forward: bool = True,
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
            {"mw", "mwss", "dh"}. Defaults to "mw".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        np.ndarray: Transform kernel for spin-spherical harmonic transform.

    Notes:
        This function adopts the Risbo Wigner d-function recursion and exploits the
        Fourier decomposition of Wigner D-functions. This involves (minor) additional
        precomputations, but is stable to effectively arbitrarily large spin numbers.
    """
    if reality and spin != 0:
        reality = False
        warn(
            "Reality acceleration only supports spin 0 fields. "
            + "Defering to complex transform."
        )

    m_start_ind = L - 1 if reality else 0
    m_dim = L if reality else 2 * L - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling)

    # Number of samples for inverse FFT over Wigner Fourier coefficients.
    if sampling.lower() == "mw":
        nsamps = 2 * len(thetas) - 1
    elif sampling.lower() == "mwss":
        nsamps = 2 * len(thetas) - 2
    elif sampling.lower() == "dh":
        nsamps = 2 * len(thetas)
    else:
        raise ValueError("Sampling in supported list [mw, mwss, dh]")

    # Compute Wigner d-functions from their Fourier decomposition.
    delta = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = np.zeros((len(thetas), L, m_dim), dtype=np.float64)

    for el in range(L):
        delta = recursions.risbo.compute_full(delta, np.pi / 2, L, el)
        m_value = np.arange(-L + 1, L)
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
    dl = np.einsum(
        "tlm,l->tlm", dl, np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    )

    # Fold in quadrature to avoid recomputation at run-time.
    if forward:
        weights = quadrature.quad_weights_transform(L, sampling, 0)
        dl = np.einsum("...tlm, ...t->...tlm", dl, weights)

    return dl


def spin_spherical_kernel_jax(
    L: int,
    spin: int = 0,
    reality: bool = False,
    sampling: str = "mw",
    forward: bool = True,
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
            {"mw", "mwss", "dh"}. Defaults to "mw".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        jnp.ndarray: Transform kernel for spin-spherical harmonic transform.

    Notes:
        This function adopts the Risbo Wigner d-function recursion and exploits the
        Fourier decomposition of Wigner D-functions. This involves (minor) additional
        precomputations, but is stable to effectively arbitrarily large spin numbers.
    """
    m_start_ind = L - 1 if reality else 0
    m_dim = L if reality else 2 * L - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling)

    # Number of samples for inverse FFT over Wigner Fourier coefficients.
    if sampling.lower() == "mw":
        nsamps = 2 * len(thetas) - 1
    elif sampling.lower() == "mwss":
        nsamps = 2 * len(thetas) - 2
    elif sampling.lower() == "dh":
        nsamps = 2 * len(thetas)
    else:
        raise ValueError("Sampling in supported list [mw, mwss, dh]")

    # Compute Wigner d-functions from their Fourier decomposition.
    delta = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl = jnp.zeros((len(thetas), L, m_dim), dtype=jnp.float64)

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

    # Fold in normalisation to avoid recomputation at run-time.
    dl = jnp.einsum(
        "tlm,l->tlm", dl, jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi))
    )

    # Fold in quadrature to avoid recomputation at run-time.
    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling)
        dl = jnp.einsum("...tlm, ...t->...tlm", dl, weights)

    return dl


def wigner_kernel(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
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
            {"mw", "mwss", "dh"}. Defaults to "mw".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        np.ndarray: Transform kernel for Wigner transform.

    Notes:
        This function adopts the Risbo Wigner d-function recursion and exploits the
        Fourier decomposition of Wigner D-functions. This involves (minor) additional
        precomputations, but is stable to effectively arbitrarily large spin numbers.
    """
    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling)

    # Number of samples for inverse FFT over Wigner Fourier coefficients.
    if sampling.lower() == "mw":
        nsamps = 2 * len(thetas) - 1
    elif sampling.lower() == "mwss":
        nsamps = 2 * len(thetas) - 2
    elif sampling.lower() == "dh":
        nsamps = 2 * len(thetas)
    else:
        raise ValueError("Sampling in supported list [mw, mwss, dh]")

    # Compute Wigner d-functions from their Fourier decomposition.
    if N <= int(L / np.log(L)):
        delta = np.zeros((len(thetas), 2 * L - 1, 2 * L - 1), dtype=np.float64)
    else:
        delta = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = np.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=np.float64)

    # Range values which need only be defined once.
    m_value = np.arange(-L + 1, L)
    n = np.arange(n_start_ind - N + 1, N)

    # If N <= L/LogL more efficient to manually compute over FFT
    for el in range(L):
        if N <= int(L / np.log(L)):
            delta = recursions.risbo.compute_full_vect(delta, thetas, L, el)
            dl[:, :, el] = np.moveaxis(delta, -1, 0)[L - 1 + n]
        else:
            delta = recursions.risbo.compute_full(delta, np.pi / 2, L, el)
            temp = np.einsum(
                "am,an,m,n->amn",
                delta,
                delta[:, L - 1 + n],
                1j ** (-m_value),
                1j ** (n),
            )
            temp = np.einsum(
                "amn,a->amn", temp, np.exp(1j * m_value * thetas[0])
            )
            temp = np.fft.irfft(temp[L - 1 :], n=nsamps, axis=0, norm="forward")
            dl[:, :, el] = np.moveaxis(temp[: len(thetas)], -1, 0)

    if forward:
        weights = quadrature.quad_weights_transform(L, sampling)
        dl = np.einsum("...ntlm, ...t->...ntlm", dl, weights)
        dl *= 2 * np.pi / (2 * N - 1)

    else:
        dl = np.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * np.arange(L) + 1) / (8 * np.pi**2),
        )

    return dl


def wigner_kernel_jax(
    L: int,
    N: int,
    reality: bool = False,
    sampling: str = "mw",
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
            {"mw", "mwss", "dh"}. Defaults to "mw".

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

    Returns:
        jnp.ndarray: Transform kernel for Wigner transform.

    Notes:
        This function adopts the Risbo Wigner d-function recursion and exploits the
        Fourier decomposition of Wigner D-functions. This involves (minor) additional
        precomputations, but is stable to effectively arbitrarily large spin numbers.
    """

    n_start_ind = N - 1 if reality else 0
    n_dim = N if reality else 2 * N - 1

    # Determine theta locations for forward vs inverse transform.
    if forward and sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        thetas = samples.thetas(2 * L, "mwss")
    else:
        thetas = samples.thetas(L, sampling)

    # Number of samples for inverse FFT over Wigner Fourier coefficients.
    if sampling.lower() == "mw":
        nsamps = 2 * len(thetas) - 1
    elif sampling.lower() == "mwss":
        nsamps = 2 * len(thetas) - 2
    elif sampling.lower() == "dh":
        nsamps = 2 * len(thetas)
    else:
        raise ValueError("Sampling in supported list [mw, mwss, dh]")

    # Compute Wigner d-functions from their Fourier decomposition.
    if N <= int(L / np.log(L)):
        delta = jnp.zeros(
            (len(thetas), 2 * L - 1, 2 * L - 1), dtype=jnp.float64
        )
        vfunc = jax.vmap(
            recursions.risbo_jax.compute_full, in_axes=(0, 0, None, None)
        )
    else:
        delta = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl = jnp.zeros((n_dim, len(thetas), L, 2 * L - 1), dtype=jnp.float64)

    # Range values which need only be defined once.
    m_value = jnp.arange(-L + 1, L)
    n = jnp.arange(n_start_ind - N + 1, N)

    # If N <= L/LogL more efficient to manually compute over FFT
    for el in range(L):
        if N <= int(L / np.log(L)):
            delta = vfunc(delta, thetas, L, el)
            dl = dl.at[:, :, el].set(jnp.moveaxis(delta, -1, 0)[L - 1 + n])
        else:
            delta = recursions.risbo_jax.compute_full(delta, jnp.pi / 2, L, el)
            temp = jnp.einsum(
                "am,an,m,n->amn",
                delta,
                delta[:, L - 1 + n],
                1j ** (-m_value),
                1j ** (n),
            )
            temp = jnp.einsum(
                "amn,a->amn", temp, jnp.exp(1j * m_value * thetas[0])
            )
            temp = jnp.fft.irfft(
                temp[L - 1 :], n=nsamps, axis=0, norm="forward"
            )
            dl = dl.at[:, :, el].set(jnp.moveaxis(temp[: len(thetas)], -1, 0))

    if forward:
        weights = quadrature_jax.quad_weights_transform(L, sampling)
        dl = jnp.einsum("...ntlm, ...t->...ntlm", dl, weights)
        dl *= 2 * jnp.pi / (2 * N - 1)

    else:
        dl = jnp.einsum(
            "...ntlm,...l->...ntlm",
            dl,
            (2 * jnp.arange(L) + 1) / (8 * jnp.pi**2),
        )

    return dl
