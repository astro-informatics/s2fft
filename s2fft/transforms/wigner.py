from functools import partial
from typing import List

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

import s2fft
from s2fft.sampling import so3_samples as samples
from s2fft.transforms import c_backend_spherical as c_sph
from s2fft.utils import torch_wrapper


def inverse(
    flmn: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
    _ssht_backend: int = 1,
) -> np.ndarray:
    r"""
    Wrapper for the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Should the user select method = "jax_ssht" they will be restricted to deployment on
    CPU using our custom JAX frontend for the SSHT C library [1]. In many
    cases this approach may be desirable to mitigate e.g. memory i/o cost.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit. For high precision beyond :math:`N \approx 8`,
            one should use `method="jax_ssht"`.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax", "jax_ssht"}.
            Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Raises:
        ValueError: Transform method not recognised.

    Returns:
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\gamma},
        n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.

    """
    if method not in _inverse_functions:
        raise ValueError(f"Method {method} not recognised.")

    if N >= 8 and method in ("numpy", "jax", "torch"):
        raise Warning("Recursive transform may provide lower precision beyond N ~ 8")

    inverse_kwargs = {
        "flmn": flmn,
        "L": L,
        "N": N,
        "L_lower": L_lower,
        "sampling": sampling,
        "reality": reality,
    }

    if method in ("jax", "numpy", "torch"):
        inverse_kwargs.update(nside=nside, precomps=precomps)

    if method == "jax_ssht":
        if sampling.lower() == "healpix":
            raise ValueError("SSHT does not support healpix sampling.")
        inverse_kwargs["_ssht_backend"] = _ssht_backend

    return _inverse_functions[method](**inverse_kwargs)


def inverse_numpy(
    flmn: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""
    Compute the inverse Wigner transform (numpy).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(NL^2)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit. Recursive transform will have lower precision
            beyond :math:`N \approx 8`.

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
    if precomps is None:
        precomps = s2fft.generate_precomputes_wigner(
            L, N, sampling, nside, False, reality, L_lower
        )
    fban = np.zeros(samples.f_shape(L, N, sampling, nside), dtype=np.complex128)

    # Copy flmn argument to avoid in-place updates being propagated back to caller
    flmn = flmn.copy()

    flmn[:, L_lower:] = np.einsum(
        "...nlm,...l->...nlm",
        flmn[:, L_lower:],
        np.sqrt((2 * np.arange(L_lower, L) + 1) / (16 * np.pi**3)),
    )

    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        fban[N - 1 + n] = (-1) ** n * s2fft.inverse_numpy(
            flmn[N - 1 + n],
            L,
            -n,
            nside,
            sampling,
            reality if n == 0 else False,
            precomps[n - n_start_ind],
            L_lower,
        )

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        f = np.fft.irfft(fban[N - 1 :], 2 * N - 1, axis=ax, norm="forward")
    else:
        f = np.fft.ifft(np.fft.ifftshift(fban, axes=ax), axis=ax, norm="forward")

    return f


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def inverse_jax(
    flmn: jnp.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""
    Compute the inverse Wigner transform (JAX).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(NL^2)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Args:
        flmn (jnp.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit. Recursive transform will have lower precision
            beyond :math:`N \approx 8`.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Signal on the sphere.

    """
    if precomps is None:
        precomps = s2fft.generate_precomputes_wigner_jax(
            L, N, sampling, nside, False, reality, L_lower
        )

    fban = jnp.zeros(samples.f_shape(L, N, sampling, nside), dtype=jnp.complex128)

    flmn = flmn.at[:, L_lower:].set(
        jnp.einsum(
            "...nlm,...l->...nlm",
            flmn[:, L_lower:],
            jnp.sqrt((2 * jnp.arange(L_lower, L) + 1) / (16 * jnp.pi**3)),
            optimize=True,
        )
    )

    n_start_ind = 0 if reality else -N + 1
    spins = jnp.arange(n_start_ind, N)

    def func(flm, spin, p0, p1, p2, p3, p4):
        precomps = [p0, p1, p2, p3, p4]
        return (-1) ** jnp.abs(spin) * s2fft.inverse_jax(
            flm, L, -spin, nside, sampling, False, precomps, False, L_lower
        )

    fban = fban.at[N - 1 + n_start_ind :].set(
        vmap(
            partial(func, p2=precomps[2][0], p3=precomps[3][0], p4=precomps[4][0]),
            in_axes=(0, 0, 0, 0),
        )(flmn[N - 1 + n_start_ind :], spins, precomps[0], precomps[1])
    )
    if reality:
        f = jnp.fft.irfft(fban[N - 1 :], 2 * N - 1, axis=0, norm="forward")
    else:
        f = jnp.fft.ifft(jnp.fft.ifftshift(fban, axes=0), axis=0, norm="forward")

    return f


inverse_torch = torch_wrapper.wrap_as_torch_function(inverse_jax)


def inverse_jax_ssht(
    flmn: jnp.ndarray,
    L: int,
    N: int,
    L_lower: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    r"""
    Compute the inverse Wigner transform (SSHT JAX).

    SSHT is a C library which implements the spin-spherical harmonic transform outlined
    in McEwen & Wiaux 2011 [1]. We make use of their python bindings for which we
    provide custom JAX frontends, hence providing support for automatic differentiation.
    Currently these transforms can only be deployed on CPU, which is a limitation of the
    SSHT C package.

    Args:
        flmn (jnp.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Returns:
        np.ndarray: Signal on the sphere.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.

    """
    flmn, fban = _inverse_norm(flmn, L, N, L_lower, sampling)
    fban = _flmn_to_fban(flmn, fban, L, N, sampling, reality, _ssht_backend)
    return _fban_to_f(fban, L, N, reality)


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
    _ssht_backend: int = 1,
) -> np.ndarray:
    r"""
    Wrapper for the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Should the user select method = "jax_ssht" they will be restricted to deployment on
    CPU using our custom JAX frontend for the SSHT C library [1]. In many cases this
    approach may be desirable to mitigate e.g. memory i/o cost.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax", "jax_ssht"}.
            Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[np.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Raises:
        ValueError: Transform method not recognised.

    Returns:
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.

    """
    if method not in _inverse_functions:
        raise ValueError(f"Method {method} not recognised.")

    if N >= 8 and method in ("numpy", "jax", "torch"):
        raise Warning("Recursive transform may provide lower precision beyond N ~ 8")

    forward_kwargs = {
        "f": f,
        "L": L,
        "N": N,
        "L_lower": L_lower,
        "sampling": sampling,
        "reality": reality,
    }

    if method in ("jax", "numpy", "torch"):
        forward_kwargs.update(nside=nside, precomps=precomps)

    if method == "jax_ssht":
        if sampling.lower() == "healpix":
            raise ValueError("SSHT does not support healpix sampling.")
        forward_kwargs["_ssht_backend"] = _ssht_backend

    return _forward_functions[method](**forward_kwargs)


def forward_numpy(
    f: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> np.ndarray:
    r"""
    Compute the forward Wigner transform (numpy).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(NL^2)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

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
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    """
    if precomps is None:
        precomps = s2fft.generate_precomputes_wigner(
            L, N, sampling, nside, True, reality, L_lower
        )
    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        fban = np.fft.rfft(np.real(f), axis=ax, norm="backward")
    else:
        fban = np.fft.fftshift(np.fft.fft(f, axis=ax, norm="backward"), axes=ax)

    fban *= 2 * np.pi / (2 * N - 1)

    if reality:
        sgn = (-1) ** abs(np.arange(-L + 1, L))

    n_start_ind = 0 if reality else -N + 1
    for n in range(n_start_ind, N):
        flmn[N - 1 + n] = (-1) ** n * s2fft.forward_numpy(
            fban[n - n_start_ind],
            L,
            -n,
            nside,
            sampling,
            reality if n == 0 else False,
            precomps[n - n_start_ind],
            L_lower,
        )
        if reality and n != 0:
            flmn[N - 1 - n] = np.conj(
                np.flip(flmn[N - 1 + n] * sgn * (-1) ** n, axis=-1)
            )
    flmn[:, L_lower:] = np.einsum(
        "...nlm,...l->...nlm",
        flmn[:, L_lower:],
        np.sqrt(4 * np.pi / (2 * np.arange(L_lower, L) + 1)),
    )
    return flmn


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def forward_jax(
    f: jnp.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""
    Compute the forward Wigner transform (JAX).

    Uses separation of variables and exploits the Price & McEwen recursion for accelerated
    and numerically stable Wiger-d on-the-fly recursions. The memory overhead for this
    function is theoretically :math:`\mathcal{O}(NL^2)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Args:
        f (jnp.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl", "healpix"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

    Returns:
        jnp.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    """
    if precomps is None:
        precomps = s2fft.generate_precomputes_wigner_jax(
            L, N, sampling, nside, True, reality, L_lower
        )

    flmn = jnp.zeros(samples.flmn_shape(L, N), dtype=jnp.complex128)

    if reality:
        fban = jnp.fft.rfft(jnp.real(f), axis=0, norm="backward")
    else:
        fban = jnp.fft.fftshift(jnp.fft.fft(f, axis=0, norm="backward"), axes=0)

    fban *= 2 * jnp.pi / (2 * N - 1)
    n_start_ind = 0 if reality else -N + 1
    spins = jnp.arange(n_start_ind, N)

    def func(fba, spin, p0, p1, p2, p3, p4):
        precomps = [p0, p1, p2, p3, p4]
        return (-1) ** jnp.abs(spin) * s2fft.forward_jax(
            fba, L, -spin, nside, sampling, False, precomps, False, L_lower
        )

    flmn = flmn.at[N - 1 + n_start_ind :].set(
        vmap(
            partial(func, p2=precomps[2][0], p3=precomps[3][0], p4=precomps[4][0]),
            in_axes=(0, 0, 0, 0),
        )(fban, spins, precomps[0], precomps[1])
    )

    if reality:
        nidx = jnp.arange(1, N)
        sgn = (-1) ** abs(jnp.arange(-L + 1, L))
        flmn = flmn.at[N - 1 - nidx].set(
            jnp.conj(
                jnp.flip(
                    jnp.einsum(
                        "nlm,m,n->nlm",
                        flmn[N - 1 + nidx],
                        sgn,
                        (-1) ** nidx,
                        optimize=True,
                    ),
                    axis=-1,
                )
            )
        )

    flmn = flmn.at[:, L_lower:].set(
        jnp.einsum(
            "...nlm,...l->...nlm",
            flmn[:, L_lower:],
            jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(L_lower, L) + 1)),
            optimize=True,
        )
    )
    return flmn


forward_torch = torch_wrapper.wrap_as_torch_function(forward_jax)


def forward_jax_ssht(
    f: jnp.ndarray,
    L: int,
    N: int,
    L_lower: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    r"""
    Compute the forward Wigner transform (SSHT JAX).

    SSHT is a C library which implements the spin-spherical harmonic transform outlined
    in McEwen & Wiaux 2011 [1]. We make use of their python bindings for which we
    provide custom JAX frontends, hence providing support for automatic differentiation.
    Currently these transforms can only be deployed on CPU, which is a limitation of the
    SSHT C package.

    Args:
        f (jnp.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

        L_lower (int, optional): Harmonic lower-bound. Transform will only be computed
            for :math:`\texttt{L_lower} \leq \ell < \texttt{L}`. Defaults to 0.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "gl"}.  Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        _ssht_backend (int, optional, experimental): Whether to default to SSHT core
            (set to 0) recursions or pick up ducc0 (set to 1) accelerated experimental
            backend. Use with caution.

    Returns:
        jnp.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    Note:
        [1] McEwen, Jason D. and Yves Wiaux. “A Novel Sampling Theorem on the Sphere.”
            IEEE Transactions on Signal Processing 59 (2011): 5876-5887.

    """
    flmn, fban = _f_to_fban(f, L, N, reality)
    flmn = _fban_to_flmn(flmn, fban, L, N, sampling, reality, _ssht_backend)
    return _reality_and_norm(flmn, L, N, L_lower, reality)


@partial(jit, static_argnums=(1, 2, 3))
def _f_to_fban(f: jnp.ndarray, L: int, N: int, reality: bool = False) -> jnp.ndarray:
    """Private function which maps from f to fban (C backend)."""
    flmn = jnp.zeros(samples.flmn_shape(L, N), dtype=jnp.complex128)

    if reality:
        fban = jnp.fft.rfft(jnp.real(f), axis=0, norm="backward")
    else:
        fban = jnp.fft.fftshift(jnp.fft.fft(f, axis=0, norm="backward"), axes=0)

    fban *= 2 * jnp.pi / (2 * N - 1)

    return flmn, fban


def _fban_to_flmn(
    flmn: jnp.ndarray,
    fban: jnp.ndarray,
    L: int,
    N: int,
    sampling: str = "mw",
    reality: bool = False,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    """Private function which maps from fban to flmn (C backend)."""
    ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())
    n_start_ind = 0 if reality else -N + 1
    func = partial(
        c_sph.ssht_forward,
        L=L,
        reality=False,
        ssht_sampling=ssht_sampling,
        _ssht_backend=_ssht_backend,
    )
    for n in range(n_start_ind, N):
        flmn = flmn.at[N - 1 + n].add(
            (-1) ** jnp.abs(n) * func(fban[int(n - n_start_ind)], spin=-n)
        )
    return flmn


@partial(jit, static_argnums=(1, 2, 3, 4))
def _reality_and_norm(
    flmn: jnp.ndarray, L: int, N: int, L_lower: int = 0, reality: bool = False
) -> jnp.ndarray:
    """Private function which maps from f to fban (C backend)."""
    if reality:
        nidx = jnp.arange(1, N)
        sgn = (-1) ** abs(jnp.arange(-L + 1, L))
        flmn = flmn.at[N - 1 - nidx].set(
            jnp.conj(
                jnp.flip(
                    jnp.einsum(
                        "nlm,m,n->nlm",
                        flmn[N - 1 + nidx],
                        sgn,
                        (-1) ** nidx,
                        optimize=True,
                    ),
                    axis=-1,
                )
            )
        )

    flmn = flmn.at[:, L_lower:].set(
        jnp.einsum(
            "...nlm,...l->...nlm",
            flmn[:, L_lower:],
            jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(L_lower, L) + 1)),
            optimize=True,
        )
    )
    return flmn


@partial(jit, static_argnums=(1, 2, 3, 4))
def _inverse_norm(
    flmn: jnp.ndarray, L: int, N: int, L_lower: int = 0, sampling: str = "mw"
):
    """Private function which normalised flmn for inverse Wigner (C backend)."""
    fban = jnp.zeros(samples.f_shape(L, N, sampling), dtype=jnp.complex128)

    flmn = flmn.at[:, L_lower:].set(
        jnp.einsum(
            "...nlm,...l->...nlm",
            flmn[:, L_lower:],
            jnp.sqrt((2 * jnp.arange(L_lower, L) + 1) / (16 * jnp.pi**3)),
            optimize=True,
        )
    )
    return flmn, fban


def _flmn_to_fban(
    flmn: jnp.ndarray,
    fban: jnp.ndarray,
    L: int,
    N: int,
    sampling: str = "mw",
    reality: bool = False,
    _ssht_backend: int = 1,
) -> jnp.ndarray:
    """Private function which maps from flmn to fban (C backend)."""
    ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())
    n_start_ind = 0 if reality else -N + 1
    func = partial(
        c_sph.ssht_inverse,
        L=L,
        reality=False,
        ssht_sampling=ssht_sampling,
        _ssht_backend=_ssht_backend,
    )
    for n in range(n_start_ind, N):
        fban = fban.at[N - 1 + n].add(
            (-1) ** jnp.abs(n) * func(flmn[N - 1 + n], spin=-n)
        )
    return fban


@partial(jit, static_argnums=(1, 2, 3))
def _fban_to_f(fban: jnp.ndarray, L: int, N: int, reality: bool = False) -> jnp.ndarray:
    """Private function which maps from fban to f (C backend)."""
    if reality:
        f = jnp.fft.irfft(fban[N - 1 :], 2 * N - 1, axis=-3, norm="forward")
    else:
        f = jnp.fft.ifft(jnp.fft.ifftshift(fban, axes=-3), axis=-3, norm="forward")
    return f


_inverse_functions = {
    "numpy": inverse_numpy,
    "jax": inverse_jax,
    "jax_ssht": inverse_jax_ssht,
    "torch": inverse_torch,
}

_forward_functions = {
    "numpy": forward_numpy,
    "jax": forward_jax,
    "jax_ssht": forward_jax_ssht,
    "torch": forward_torch,
}
