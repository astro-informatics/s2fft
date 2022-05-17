import numpy as np
from s2fft.precompute.construct_legendre_matrix import (
    load_legendre_matrix,
)

from jax import jit
from jax import device_put
from functools import partial
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

import s2fft.logs as lg

lg.setup_logging()


def forward_precompute(
    f: np.ndarray,
    L: int = 4,
    legendre_kernel: np.ndarray = None,
    device: str = "cpu",
    spin: int = 0,
    save_dir: str = "../../.matrices",
):
    r"""Compute forward spherical harmonic transform via precompute.

    Note:
        Only MW sampling supported at present.

    Args:
        f (np.ndarray): Signal on sphere (shape: L, 2L-1).

        L (int): Harmonic band-limit.

        legendre_kernel (np.ndarray): Legendre transform kernel.

        device (str, optional): Evaluate on "cpu" or "gpu".

        spin (int, optional): Harmonic spin. Defaults to 0.

        save_dir (str, optional): Directory to save legendre kernels.

    Returns:
        np.ndarray: 2D spherical harmonic coefficients, i.e. L x (2L-1).
    """
    if legendre_kernel is None:
        kernel = load_legendre_matrix(
            L=L, forward=True, spin=spin, save_dir=save_dir
        )
    else:
        kernel = legendre_kernel

    lg.debug_log(
        "Running precompute forward harmonic transform for L={} on {}".format(L, device)
    )

    if device == "cpu":
        return forward_transform_cpu(f, kernel, L)
    elif device == "gpu":
        forward_jit = jit(forward_transform_gpu, static_argnums=(2,))
        return forward_jit(device_put(f), device_put(kernel), L)
    else:
        raise ValueError("Device not recognised.")


def forward_transform_cpu(f: np.ndarray, legendre_kernel: np.ndarray, L: int):
    r"""Compute the forward spherical harmonic transform via precompute
    (vectorized implementation).

    Note:
        Only MW sampling supported at present.

    Args:
        f (np.ndarray): Signal on sphere (shape: L, 2L-1).

        legendre_kernel (np.ndarray): Legendre transform kernel.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 2D spherical harmonic coefficients, i.e. L x (2L-1).
    """
    fm = np.fft.fft(f)
    flm = np.einsum("lmi, im->lm", legendre_kernel, fm)
    return np.fft.fftshift(flm, axes=1)


@partial(jit, static_argnums=(2,))
def forward_transform_gpu(f: jnp.ndarray, legendre_kernel: jnp.ndarray, L: int):
    r"""Compute the forward spherical harmonic transform via precompute (JAX
    implementation).

    Args:
        f (jnp.ndarray): Signal on sphere (shape: L, 2L-1).

        legendre_kernel (jnp.ndarray): Legendre transform kernel.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 2D spherical harmonic coefficients, i.e. L x (2L-1).
    """
    fm = jnp.fft.fft(f)
    flm = jnp.einsum("lmi, im->lm", legendre_kernel, fm, optimize=True)
    return jnp.fft.fftshift(flm, axes=1)


def inverse_precompute(
    flm: np.ndarray,
    L: int = 4,
    legendre_kernel: np.ndarray = None,
    device: str = "cpu",
    spin: int = 0,
    save_dir: str = "../../.matrices",
):
    r"""Compute the inverse spherical harmonic transform via precompute.

    Args:
        flm (np.ndarray): Harmonic coefficients (shape: L(2L-1)).

        L (int): Harmonic band-limit.

        legendre_kernel (np.ndarray): Legendre transform kernel.

        device (str, optional): Evaluate on "cpu" or "gpu".

        spin (int, optional): Harmonic spin. Defaults to 0.

        save_dir (str, optional): Dorectory to save legendre kernels.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [L, 2L-1].
    """
    if legendre_kernel is None:
        kernel = load_legendre_matrix(
            L=L, forward=False, spin=spin, save_dir=save_dir
        )
    else:
        kernel = legendre_kernel

    lg.debug_log(
        "Running precompute inverse harmonic transform for L={} on {}".format(L, device)
    )

    if device == "cpu":
        return inverse_transform_cpu(flm, kernel, L)
    elif device == "gpu":
        inverse_jit = jit(inverse_transform_gpu, static_argnums=(2,))
        return inverse_jit(device_put(flm), device_put(kernel), L)
    else:
        raise ValueError("Device not recognised.")


def inverse_transform_cpu(flm: np.ndarray, legendre_kernel: np.ndarray, L: int):
    r"""Compute the forward spherical harmonic transform via precompute (vectorized
    implementation).

    Args:
        flm (np.ndarray): Harmonic coefficients (shape: L(2L-1)).

        legendre_kernel (np.ndarray): Legendre transform kernel.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [L, 2L-1].
    """
    flm_shift = np.fft.ifftshift(flm, axes=1)
    fm = np.einsum("lmi, lm->im", legendre_kernel, flm_shift)
    return fm.shape[1] * np.fft.ifft(fm)


@partial(jit, static_argnums=(2,))
def inverse_transform_gpu(flm: jnp.ndarray, legendre_kernel: jnp.ndarray, L: int):
    r"""Compute the inverse spherical harmonic transform via precompute (JAX
    implementation).

    Args:
        flm (jnp.ndarray): Harmonic coefficients (shape: L(2L-1)).

        legendre_kernel (jnp.ndarray): Legendre transform kernel.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Pixel-space coefficients with shape [L, 2L-1].
    """
    flm_shift = jnp.fft.ifftshift(flm, axes=1)
    fm = jnp.einsum("lmi, lm->im", legendre_kernel, flm_shift, optimize=True)
    return float(2 * L - 1) * jnp.fft.ifft(fm)
