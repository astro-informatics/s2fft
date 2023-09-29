from jax import jit, pmap, local_device_count


import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from typing import List
import s2fft
from s2fft.sampling import so3_samples as samples


def inverse(
    flmn: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Wrapper for the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
    \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
    :math:`zyz` Euler convention, in order to simplify indexing for internal use.
    For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
    :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
    :math:`\alpha` with :math:`\phi`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Azimuthal band-limit.

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
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\gamma},
        n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    if method == "numpy":
        return inverse_numpy(flmn, L, N, nside, sampling, reality, precomps, L_lower)
    elif method == "jax":
        return inverse_jax(
            flmn, L, N, nside, sampling, reality, precomps, spmd, L_lower
        )
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


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
    r"""Compute the inverse Wigner transform (numpy).

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

        N (int): Azimuthal band-limit.

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
    if precomps is None:
        precomps = s2fft.generate_precomputes_wigner(
            L, N, sampling, nside, False, reality, L_lower
        )
    fban = np.zeros(samples.f_shape(L, N, sampling, nside), dtype=np.complex128)

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


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7, 8))
def inverse_jax(
    flmn: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""Compute the inverse Wigner transform (numpy).

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

        N (int): Azimuthal band-limit.

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

    def spherical_loop(n, args):
        fban, flmn, lrenorm, vsign, spins = args
        fban = fban.at[n].add(
            (-1) ** spins[n]
            * s2fft.inverse_jax(
                flmn[n],
                L,
                -spins[n],
                nside,
                sampling,
                False,
                [
                    lrenorm[n],
                    vsign[n],
                    precomps[2][0],
                    precomps[3][0],
                    precomps[4][0],
                ],
                False,
                L_lower,
            )
        )
        return fban, flmn, lrenorm, vsign, spins

    if spmd:
        # TODO: Generalise this to optional device counts.
        ndevices = local_device_count()
        opsdevice = int(N / ndevices) if reality else int((2 * N - 1) / ndevices)

        def eval_spherical_loop(fban, flmn, lrenorm, vsign, spins):
            return lax.fori_loop(
                0,
                opsdevice,
                spherical_loop,
                (fban, flmn, lrenorm, vsign, spins),
            )[0]

        # Reshape inputs
        spmd_shape = (ndevices, opsdevice)
        lrenorm = precomps[0].reshape(spmd_shape + precomps[0].shape[1:])
        vsign = precomps[1].reshape(spmd_shape + precomps[1].shape[1:])
        vin = flmn[N - 1 + n_start_ind :].reshape(spmd_shape + flmn.shape[1:])
        vout = fban[N - 1 + n_start_ind :].reshape(spmd_shape + fban.shape[1:])
        spins = spins.reshape(spmd_shape)

        fban = fban.at[N - 1 + n_start_ind :].add(
            pmap(eval_spherical_loop, in_axes=(0, 0, 0, 0, 0))(
                vout, vin, lrenorm, vsign, spins
            ).reshape((ndevices * opsdevice,) + fban.shape[1:])
        )
    else:
        opsdevice = N if reality else 2 * N - 1
        fban = fban.at[N - 1 + n_start_ind :].add(
            lax.fori_loop(
                0,
                opsdevice,
                spherical_loop,
                (
                    fban[N - 1 + n_start_ind :],
                    flmn[N - 1 + n_start_ind :],
                    precomps[0],
                    precomps[1],
                    spins,
                ),
            )[0]
        )

    if reality:
        fban = fban.at[: N - 1].set(jnp.flip(jnp.conj(fban[N:]), axis=0))
    fban = jnp.conj(jnp.fft.ifftshift(fban, axes=0))
    f = jnp.conj(jnp.fft.fft(fban, axis=0, norm="backward"))

    return f


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    method: str = "numpy",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> np.ndarray:
    r"""Wrapper for the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

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
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    Note:
        The single-program multiple-data (SPMD) optional variable determines whether
        the transform is run over a single device or all available devices. For very low
        harmonic bandlimits L this is inefficient as the I/O overhead for communication
        between devices is noticable, however as L increases one will asymptotically
        recover acceleration by the number of devices.
    """
    if method == "numpy":
        return forward_numpy(f, L, N, nside, sampling, reality, precomps, L_lower)
    elif method == "jax":
        return forward_jax(f, L, N, nside, sampling, reality, precomps, spmd, L_lower)
    else:
        raise ValueError(
            f"Implementation {method} not recognised. Should be either numpy or jax."
        )


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
    r"""Compute the forward Wigner transform (numpy).

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


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7, 8))
def forward_jax(
    f: jnp.ndarray,
    L: int,
    N: int,
    nside: int = None,
    sampling: str = "mw",
    reality: bool = False,
    precomps: List = None,
    spmd: bool = False,
    L_lower: int = 0,
) -> jnp.ndarray:
    r"""Compute the forward Wigner transform (JAX).

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
            {"mw", "mwss", "dh", "healpix"}.  Defaults to "mw".

        method (str, optional): Execution mode in {"numpy", "jax"}. Defaults to "numpy".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.  Defaults to
            False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

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

    if reality:
        sgn = (-1) ** abs(jnp.arange(-L + 1, L))

    n_start_ind = 0 if reality else -N + 1
    spins = jnp.arange(n_start_ind, N)

    def spherical_loop(n, args):
        flmn, fban, lrenorm, vsign, spins = args
        flmn = flmn.at[n].add(
            (-1) ** spins[n]
            * s2fft.forward_jax(
                fban[n],
                L,
                -spins[n],
                nside,
                sampling,
                False,
                [
                    lrenorm[n],
                    vsign[n],
                    precomps[2][0],
                    precomps[3][0],
                    precomps[4][0],
                ],
                False,
                L_lower,
            )
        )
        return flmn, fban, lrenorm, vsign, spins

    if spmd:
        # TODO: Generalise this to optional device counts.
        ndevices = local_device_count()
        opsdevice = int(N / ndevices) if reality else int((2 * N - 1) / ndevices)

        def eval_spherical_loop(fban, flmn, lrenorm, vsign, spins):
            return lax.fori_loop(
                0,
                opsdevice,
                spherical_loop,
                (fban, flmn, lrenorm, vsign, spins),
            )[0]

        # Reshape inputs
        spmd_shape = (ndevices, opsdevice)
        lrenorm = precomps[0].reshape(spmd_shape + precomps[0].shape[1:])
        vsign = precomps[1].reshape(spmd_shape + precomps[1].shape[1:])
        fban = fban.reshape(spmd_shape + fban.shape[1:])
        vout = flmn[N - 1 + n_start_ind :].reshape(spmd_shape + flmn.shape[1:])
        spins = spins.reshape(spmd_shape)

        flmn = flmn.at[N - 1 + n_start_ind :].add(
            pmap(eval_spherical_loop, in_axes=(0, 0, 0, 0, 0))(
                vout, fban, lrenorm, vsign, spins
            ).reshape((ndevices * opsdevice,) + flmn.shape[1:])
        )
    else:
        opsdevice = N if reality else 2 * N - 1
        flmn = flmn.at[N - 1 + n_start_ind :].add(
            lax.fori_loop(
                0,
                opsdevice,
                spherical_loop,
                (
                    flmn[N - 1 + n_start_ind :],
                    fban,
                    precomps[0],
                    precomps[1],
                    spins,
                ),
            )[0]
        )

    # Fill out Wigner coefficients steerability of real signals
    for n in range(n_start_ind, N):
        if reality and n != 0:
            flmn = flmn.at[N - 1 - n].set(
                jnp.conj(jnp.flip(flmn[N - 1 + n] * sgn * (-1) ** n, axis=-1))
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
