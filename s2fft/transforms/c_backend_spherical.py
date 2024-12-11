from functools import partial

import healpy
import jax.numpy as jnp
import numpy as np

# C backend functions for which to provide JAX frontend.
import pyssht
from jax import core, custom_vjp
from jax.interpreters import ad

from s2fft.sampling import reindex
from s2fft.utils import iterative_refinement, quadrature_jax


@custom_vjp
def ssht_inverse(
    flm: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    r"""
    Compute the inverse spin-spherical harmonic transform (SSHT JAX).

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
    flm_1d = reindex.flm_2d_to_1d_fast(flm, L)
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    return jnp.array(
        pyssht.inverse(
            np.array(flm_1d),
            L,
            spin,
            Method=sampling_str[ssht_sampling],
            Reality=reality,
            backend=_backend,
        )
    )


def _ssht_inverse_fwd(
    flm: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
):
    """Private function which implements the forward pass for inverse jax_ssht."""
    res = ([], L, spin, reality, ssht_sampling, _ssht_backend)
    return ssht_inverse(flm, L, spin, reality, ssht_sampling, _ssht_backend), res


def _ssht_inverse_bwd(res, f):
    """Private function which implements the backward pass for inverse jax_ssht."""
    _, L, spin, reality, ssht_sampling, _ssht_backend = res
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    if ssht_sampling < 2:  # MW or MWSS sampling
        flm = jnp.array(
            np.conj(
                pyssht.inverse_adjoint(
                    np.conj(np.array(f)),
                    L,
                    spin,
                    Method=sampling_str[ssht_sampling],
                    Reality=reality,
                    backend=_backend,
                )
            )
        )
    else:  # DH or GL samping
        quad_weights = quadrature_jax.quad_weights_transform(
            L, sampling_str[ssht_sampling].lower()
        )
        f = jnp.einsum("tp,t->tp", f, 1 / quad_weights, optimize=True)
        flm = jnp.array(
            np.conj(
                pyssht.forward(
                    np.conj(np.array(f)),
                    L,
                    spin,
                    Method=sampling_str[ssht_sampling],
                    Reality=reality,
                    backend=_backend,
                )
            )
        )

    flm_out = reindex.flm_1d_to_2d_fast(flm, L)

    if reality:
        m_conj = (-1) ** (jnp.arange(1, L) % 2)
        flm_out = flm_out.at[..., L:].add(
            jnp.flip(m_conj * jnp.conj(flm_out[..., : L - 1]), axis=-1)
        )
        flm_out = flm_out.at[..., : L - 1].set(0)

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
    r"""
    Compute the forward spin-spherical harmonic transform (SSHT JAX).

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
    flm = jnp.array(
        pyssht.forward(
            np.array(f),
            L,
            spin,
            Method=sampling_str[ssht_sampling],
            Reality=reality,
            backend=_backend,
        )
    )
    return reindex.flm_1d_to_2d_fast(flm, L)


def _ssht_forward_fwd(
    f: jnp.ndarray,
    L: int,
    spin: int = 0,
    reality: bool = False,
    ssht_sampling: int = 0,
    _ssht_backend: int = 1,
):
    """Private function which implements the forward pass for forward jax_ssht."""
    res = ([], L, spin, reality, ssht_sampling, _ssht_backend)
    return ssht_forward(f, L, spin, reality, ssht_sampling, _ssht_backend), res


def _ssht_forward_bwd(res, flm):
    """Private function which implements the backward pass for forward jax_ssht."""
    _, L, spin, reality, ssht_sampling, _ssht_backend = res
    sampling_str = ["MW", "MWSS", "DH", "GL"]
    _backend = "SSHT" if _ssht_backend == 0 else "ducc0"
    flm_1d = reindex.flm_2d_to_1d_fast(flm, L)

    if ssht_sampling < 2:  # MW or MWSS sampling
        f = jnp.array(
            np.conj(
                pyssht.forward_adjoint(
                    np.conj(np.array(flm_1d)),
                    L,
                    spin,
                    Method=sampling_str[ssht_sampling],
                    Reality=reality,
                    backend=_backend,
                )
            )
        )
    else:  # DH or GL sampling
        quad_weights = quadrature_jax.quad_weights_transform(
            L, sampling_str[ssht_sampling].lower()
        )
        f = jnp.array(
            np.conj(
                pyssht.inverse(
                    np.conj(np.array(flm_1d)),
                    L,
                    spin,
                    Method=sampling_str[ssht_sampling],
                    Reality=reality,
                    backend=_backend,
                )
            )
        )
        f = jnp.einsum("tp,t->tp", f, quad_weights, optimize=True)

    return f, None, None, None, None, None


# Link JAX gradients for C backend functions
ssht_inverse.defvjp(_ssht_inverse_fwd, _ssht_inverse_bwd)
ssht_forward.defvjp(_ssht_forward_fwd, _ssht_forward_bwd)


def _complex_dtype(real_dtype):
    """
    Get complex datatype corresponding to a given real datatype.

    Derived from https://github.com/jax-ml/jax/blob/1471702adc28/jax/_src/lax/fft.py#L92

    Original license:

    Copyright 2019 The JAX Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    return (np.zeros((), real_dtype) + np.zeros((), np.complex64)).dtype


def _real_dtype(complex_dtype):
    """
    Get real datatype corresponding to a given complex datatype.

    Derived from https://github.com/jax-ml/jax/blob/1471702adc28/jax/_src/lax/fft.py#L93

    Original license:

    Copyright 2019 The JAX Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    return np.finfo(complex_dtype).dtype


def _healpy_map2alm_impl(f: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    return jnp.array(healpy.map2alm(np.array(f), lmax=L - 1, iter=0))


def _healpy_map2alm_abstract_eval(
    f: core.ShapedArray, L: int, nside: int
) -> core.ShapedArray:
    return core.ShapedArray(shape=(L * (L + 1) // 2,), dtype=_complex_dtype(f.dtype))


def _healpy_map2alm_transpose(dflm: jnp.ndarray, L: int, nside: int):
    scale_factors = (
        jnp.concatenate((jnp.ones(L), 2 * jnp.ones(L * (L - 1) // 2)))
        * (3 * nside**2)
        / jnp.pi
    )
    return (jnp.conj(healpy_alm2map(jnp.conj(dflm) / scale_factors, L, nside)),)


_healpy_map2alm_p = core.Primitive("healpy_map2alm")
_healpy_map2alm_p.def_impl(_healpy_map2alm_impl)
_healpy_map2alm_p.def_abstract_eval(_healpy_map2alm_abstract_eval)
ad.deflinear(_healpy_map2alm_p, _healpy_map2alm_transpose)


def healpy_map2alm(f: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    """
    JAX wrapper for healpy map2alm function (forward spherical harmonic transform).

    This wrapper will return the spherical harmonic coefficients as a one dimensional
    array using HEALPix (ring-ordered) indexing. To instead return a two-dimensional
    array of harmonic coefficients use :py:func:`healpy_forward`.

    Args:
        f (jnp.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit. Equivalent to `lmax + 1` in healpy.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: Harmonic coefficients of signal f.

    """
    return _healpy_map2alm_p.bind(f, L=L, nside=nside)


def _healpy_alm2map_impl(flm: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    return jnp.array(healpy.alm2map(np.array(flm), lmax=L - 1, nside=nside))


def _healpy_alm2map_abstract_eval(
    flm: core.ShapedArray, L: int, nside: int
) -> core.ShapedArray:
    return core.ShapedArray(shape=(12 * nside**2,), dtype=_real_dtype(flm.dtype))


def _healpy_alm2map_transpose(df: jnp.ndarray, L: int, nside: int) -> tuple:
    scale_factors = (
        jnp.concatenate((jnp.ones(L), 2 * jnp.ones(L * (L - 1) // 2)))
        * (3 * nside**2)
        / jnp.pi
    )
    # Scale factor above includes the inverse quadrature weight given by
    # (12 * nside**2) / (4 * jnp.pi) = (3 * nside**2) / jnp.pi
    # and also a factor of 2 for m>0 to account for the negative m.
    # See explanation in this issue comment:
    # https://github.com/astro-informatics/s2fft/issues/243#issuecomment-2500951488
    return (scale_factors * jnp.conj(healpy_map2alm(jnp.conj(df), L, nside)),)


_healpy_alm2map_p = core.Primitive("healpy_alm2map")
_healpy_alm2map_p.def_impl(_healpy_alm2map_impl)
_healpy_alm2map_p.def_abstract_eval(_healpy_alm2map_abstract_eval)
ad.deflinear(_healpy_alm2map_p, _healpy_alm2map_transpose)


def healpy_alm2map(flm: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    """
    JAX wrapper for healpy alm2map function (inverse spherical harmonic transform).

    This wrapper assumes the passed spherical harmonic coefficients are a one
    dimensional array using HEALPix (ring-ordered) indexing. To instead pass a
    two-dimensional array of harmonic coefficients use :py:func:`healpy_inverse`.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit. Equivalent to `lmax + 1` in healpy.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: Signal on the sphere.

    """
    return _healpy_alm2map_p.bind(flm, L=L, nside=nside)


def healpy_forward(f: jnp.ndarray, L: int, nside: int, iter: int = 3) -> jnp.ndarray:
    r"""
    Compute the forward scalar spherical harmonic transform (healpy JAX).

    HEALPix is a C++ library which implements the scalar spherical harmonic transform
    outlined in [1]. We make use of their healpy python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation.
    Currently these transforms can only be deployed on CPU, which is a limitation of the
    C++ library.

    Args:
        f (jnp.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        iter (int, optional): Number of subiterations (iterative refinement steps) for
            healpy. Note that iterations increase the precision of the forward transform
            as an inverse of inverse transform, but with a linear increase in
            computational cost. Between 2 and 3 iterations is a good compromise.

    Returns:
        jnp.ndarray: Harmonic coefficients of signal f.

    Note:
        [1] Gorski, Krzysztof M., et al. "HEALPix: A framework for high-resolution
        discretization and fast analysis of data distributed on the sphere." The
        Astrophysical Journal 622.2 (2005): 759

    """
    flm = iterative_refinement.forward_with_iterative_refinement(
        f=f,
        n_iter=iter,
        forward_function=partial(healpy_map2alm, L=L, nside=nside),
        backward_function=partial(healpy_alm2map, L=L, nside=nside),
    )
    return reindex.flm_hp_to_2d_fast(flm, L)


def healpy_inverse(flm: jnp.ndarray, L: int, nside: int) -> jnp.ndarray:
    r"""
    Compute the inverse scalar real spherical harmonic transform (HEALPix JAX).

    HEALPix is a C++ library which implements the scalar spherical harmonic transform
    outlined in [1]. We make use of their healpy python bindings for which we provide
    custom JAX frontends, hence providing support for automatic differentiation.
    Currently these transforms can only be deployed on CPU, which is a limitation of the
    C++ library.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: Signal on the sphere.

    Note:
        [1] Gorski, Krzysztof M., et al. "HEALPix: A framework for high-resolution
        discretization and fast analysis of data distributed on the sphere." The
        Astrophysical Journal 622.2 (2005): 759

    """
    flm = reindex.flm_2d_to_hp_fast(flm, L)
    return healpy_alm2map(flm, L, nside)
