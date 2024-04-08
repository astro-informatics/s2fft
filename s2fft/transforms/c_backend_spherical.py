from jax import custom_vjp
import numpy as np
import jax.numpy as jnp
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature_jax

# C backend functions for which to provide JAX frontend.
import pyssht
import healpy


@custom_vjp
def ssht_inverse(
    flm: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    r"""Compute the inverse spin-spherical harmonic transform (SSHT JAX).

    SSHT is a C library which implements the spin-spherical harmonic transform outlined in
    McEwen & Wiaux 2011 [1]. We make use of their python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation. Currently
    these transforms can only be deployed on CPU, which is a limitation of the SSHT C package.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        ssht_sampling (int, optional): Sampling scheme. Supported sampling schemes include
            {"mw" = 0, "mwss" = 1, "dh" = 2, "gl" = 3}.  Defaults to "mw" = 0.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Returns:
        jnp.ndarray: Signal on the sphere.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.
    """
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    flm_1d = samples.flm_2d_to_1d(flm, L)
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    return pyssht.inverse(
        flm_1d,
        L,
        spin,
        Method=sampling_str[ssht_sampling],
        Reality=reality,
        backend=_backend,
    )


def _ssht_inverse_fwd(
    flm: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
):
    """Private function which implements the forward pass for inverse jax_ssht"""
    res = ([], L, spin, reality, ssht_sampling, _ssht_backend)
    return ssht_inverse(flm, L, spin, reality, ssht_sampling, _ssht_backend), res


def _ssht_inverse_bwd(res, f):
    """Private function which implements the backward pass for inverse jax_ssht"""
    _, L, spin, reality, ssht_sampling, _ssht_backend = res
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    if ssht_sampling < 2:  # MW or MWSS sampling
        flm = np.conj(
            pyssht.inverse_adjoint(
                np.conj(f),
                L,
                spin,
                Method=sampling_str[ssht_sampling],
                Reality=reality,
                backend=_backend,
            )
        )
    else:  # DH or GL samping
        quad_weights = quadrature_jax.quad_weights_transform(
            L, sampling_str[ssht_sampling].lower()
        )
        f = jnp.einsum("tp,t->tp", f, 1 / quad_weights, optimize=True)
        flm = np.conj(
            pyssht.forward(
                np.conj(f),
                L,
                spin,
                Method=sampling_str[ssht_sampling],
                Reality=reality,
                backend=_backend,
            )
        )

    flm_out = samples.flm_1d_to_2d(flm, L)

    if reality:
        m_conj = (-1) ** (np.arange(1, L) % 2)
        flm_out[..., L:] += np.flip(m_conj * np.conj(flm_out[..., : L - 1]), axis=-1)
        flm_out[..., : L - 1] = 0

    return flm_out, None, None, None, None, None


@custom_vjp
def ssht_forward(
    f: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    r"""Compute the forward spin-spherical harmonic transform (SSHT JAX).

    SSHT is a C library which implements the spin-spherical harmonic transform outlined in
    McEwen & Wiaux 2011 [1]. We make use of their python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation. Currently
    these transforms can only be deployed on CPU, which is a limitation of the SSHT C package.

    Args:
        f (jnp.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        ssht_sampling (int, optional): Sampling scheme. Supported sampling schemes include
            {"mw" = 0, "mwss" = 1, "dh" = 2, "gl" = 3}.  Defaults to "mw" = 0.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Returns:
        jnp.ndarray: Harmonic coefficients of signal f.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.
    """
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    flm = pyssht.forward(
        np.array(f),
        L,
        spin,
        Method=sampling_str[ssht_sampling],
        Reality=reality,
        backend=_backend,
    )
    return samples.flm_1d_to_2d(flm, L)


def _ssht_forward_fwd(
    f: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
):
    """Private function which implements the forward pass for forward jax_ssht"""
    res = ([], L, spin, reality, ssht_sampling, _ssht_backend)
    return ssht_forward(f, L, spin, reality, ssht_sampling, _ssht_backend), res


def _ssht_forward_bwd(res, flm):
    """Private function which implements the backward pass for forward jax_ssht"""
    _, L, spin, reality, ssht_sampling, _ssht_backend = res
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    flm_1d = samples.flm_2d_to_1d(flm, L)

    if ssht_sampling < 2:  # MW or MWSS sampling
        f = np.conj(
            pyssht.forward_adjoint(
                np.conj(flm_1d),
                L,
                spin,
                Method=sampling_str[ssht_sampling],
                Reality=reality,
                backend=_backend,
            )
        )
    else:  # DH or GL sampling
        quad_weights = quadrature_jax.quad_weights_transform(
            L, sampling_str[ssht_sampling].lower()
        )
        f = np.conj(
            pyssht.inverse(
                np.conj(flm_1d),
                L,
                spin,
                Method=sampling_str[ssht_sampling],
                Reality=reality,
                backend=_backend,
            )
        )
        f = jnp.einsum("tp,t->tp", f, quad_weights, optimize=True)

    return f, None, None, None, None, None


@custom_vjp
def healpy_inverse(flm: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    r"""Compute the inverse scalar real spherical harmonic transform (HEALPix JAX).

    HEALPix is a C++ library which implements the scalar spherical harmonic transform
    outlined in [1]. We make use of their healpy python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation. Currently
    these transforms can only be deployed on CPU, which is a limitation of the C++ library.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

    Returns:
        jnp.ndarray: Signal on the sphere.

    Note:
        [1] Gorski, Krzysztof M., et al. "HEALPix: A framework for high-resolution
        discretization and fast analysis of data distributed on the sphere." The
        Astrophysical Journal 622.2 (2005): 759
    """
    flm = samples.flm_2d_to_hp(flm, L)
    f = healpy.alm2map(flm, lmax=L - 1, nside=nside)
    return f


def _healpy_inverse_fwd(flm: jnp.ndarray, L: int, nside: int):
    """Private function which implements the forward pass for inverse jax_healpy"""
    res = ([], L, nside)
    return healpy_inverse(flm, L, nside), res


def _healpy_inverse_bwd(res, f):
    """Private function which implements the backward pass for inverse jax_healpy"""
    _, L, nside = res
    f_new = f * (12 * nside**2) / (4 * np.pi)
    flm_out = np.conj(healpy.map2alm(np.conj(f_new), lmax=L - 1, iter=0))
    # iter MUST be zero otherwise gradient propagation is incorrect (JDM).
    flm_out = samples.flm_hp_to_2d(flm_out, L)
    m_conj = (-1) ** (np.arange(1, L) % 2)
    flm_out[..., L:] += np.flip(m_conj * np.conj(flm_out[..., : L - 1]), axis=-1)
    flm_out[..., : L - 1] = 0

    return flm_out, None, None


@custom_vjp
def healpy_forward(f: jnp.ndarray, L: int, nside: int, iter: int = 3) -> jnp.ndarray:
    r"""Compute the forward scalar spherical harmonic transform (healpy JAX).

    HEALPix is a C++ library which implements the scalar spherical harmonic transform
    outlined in [1]. We make use of their healpy python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation. Currently
    these transforms can only be deployed on CPU, which is a limitation of the C++ library.

    Args:
        f (jnp.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        iter (int, optional): Number of subiterations for healpy. Note that iterations
            increase the precision of the forward transform, but reduce the accuracy of
            the gradient pass. Between 2 and 3 iterations is a good compromise.

    Returns:
        jnp.ndarray: Harmonic coefficients of signal f.

    Note:
        [1] Gorski, Krzysztof M., et al. "HEALPix: A framework for high-resolution
        discretization and fast analysis of data distributed on the sphere." The
        Astrophysical Journal 622.2 (2005): 759
    """
    flm = healpy.map2alm(np.array(f), lmax=L - 1, iter=iter)
    return samples.flm_hp_to_2d(flm, L)


def _healpy_forward_fwd(f: jnp.ndarray, L: int, nside: int, iter: int = 3):
    """Private function which implements the forward pass for forward jax_healpy"""
    res = ([], L, nside, iter)
    return healpy_forward(f, L, nside, iter), res


def _healpy_forward_bwd(res, flm):
    """Private function which implements the backward pass for forward jax_healpy"""
    _, L, nside, _ = res
    flm_new = samples.flm_2d_to_hp(np.array(flm), L)
    f = np.conj(healpy.alm2map(np.conj(flm_new), lmax=L - 1, nside=nside))
    return f * (4 * np.pi) / (12 * nside**2), None, None, None


# Link JAX gradients for C backend functions
ssht_inverse.defvjp(_ssht_inverse_fwd, _ssht_inverse_bwd)
ssht_forward.defvjp(_ssht_forward_fwd, _ssht_forward_bwd)
healpy_inverse.defvjp(_healpy_inverse_fwd, _healpy_inverse_bwd)
healpy_forward.defvjp(_healpy_forward_fwd, _healpy_forward_bwd)
