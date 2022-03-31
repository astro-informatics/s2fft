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


def forward_precompute(f, L=4, legendre_kernel=None, device="cpu", spin=0):
    """Computes the forward spherical harmonic transform via precompute

    Args:

        f (np.ndarray): Pixel-space signal (shape: L, 2*L-1)
        L (int): Angular bandlimit
        legendre_kernel (np.ndarray): Legendre transform kernel
        device (str): Evaluate on "cpu" or "gpu"
        spin (int): Spin of the transform to consider

    Returns:

        Oversampled spherical harmonic coefficients i.e. L*(2*L-1)
        coefficients indexed by [-L < n < L].
    """
    if legendre_kernel is None:
        kernel = load_legendre_matrix(L=L, direction="forward", spin=spin)
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


def forward_transform_cpu(f, legendre_kernel, L):
    """Computes the NumPy forward spherical harmonic transform via precompute

    Args:

        f (np.ndarray): Pixel-space signal (shape: L, 2*L-1)
        legendre_kernel (np.ndarray): Legendre transform kernel
        L (int): Angular bandlimit

    Returns:

        Oversampled spherical harmonic coefficients i.e. L*(2*L-1)
        coefficients indexed by [-L < n < L].
    """
    fm = np.fft.fft(f)
    flm = np.einsum("lmi, im->lm", legendre_kernel, fm)
    return np.ravel(np.fft.fftshift(flm, axes=1))


@partial(jit, static_argnums=(2,))
def forward_transform_gpu(f, legendre_kernel, L):
    """Computes the JAX forward spherical harmonic transform via precompute

    Args:

        f (np.ndarray): Pixel-space signal (shape: L, 2*L-1)
        legendre_kernel (np.ndarray): Legendre transform kernel
        L (static_int): Angular bandlimit

    Returns:

        Oversampled spherical harmonic coefficients i.e. L*(2*L-1)
        coefficients indexed by [-L < n < L].
    """
    fm = jnp.fft.fft(f)
    flm = jnp.einsum("lmi, im->lm", legendre_kernel, fm, optimize=True)
    return jnp.ravel(jnp.fft.fftshift(flm, axes=1))


def inverse_precompute(flm, L=4, legendre_kernel=None, device="cpu", spin=0):
    """Computes the inverse spherical harmonic transform via precompute

    Args:

        flm (np.ndarray): Harmonic-space signal (shape: L*(2*L-1))
        L (int): Angular bandlimit
        legendre_kernel (np.ndarray): Legendre transform kernel
        device (str): Evaluate on "cpu" or "gpu"
        spin (int): Spin of the transform to consider

    Returns:

        Pixel-space coefficients with shape [L, 2*L-1].
    """
    if legendre_kernel is None:
        kernel = load_legendre_matrix(L=L, direction="inverse", spin=spin)
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


def inverse_transform_cpu(flm, legendre_kernel, L):
    """Computes the NumPy forward spherical harmonic transform via precompute

    Args:

        flm (np.ndarray): Harmonic-space signal (shape: L*(2*L-1))
        legendre_kernel (np.ndarray): Legendre transform kernel
        L (int): Angular bandlimit

    Returns:

        Pixel-space coefficients with shape [L, 2*L-1].
    """
    flm_shift = np.fft.ifftshift(np.reshape(flm, (L, 2 * L - 1)), axes=1)
    fm = np.einsum("lmi, lm->im", legendre_kernel, flm_shift)
    return fm.shape[1] * np.fft.ifft(fm)


@partial(jit, static_argnums=(2,))
def inverse_transform_gpu(flm, legendre_kernel, L):
    """Computes the JAX inverse spherical harmonic transform via precompute

    Args:

        flm (np.ndarray): Harmonic-space signal (shape: L*(2*L-1))
        legendre_kernel (np.ndarray): Legendre transform kernel
        L (int): Angular bandlimit

    Returns:

        Pixel-space coefficients with shape [L, 2*L-1].
    """
    flm_shift = jnp.fft.ifftshift(jnp.reshape(flm, (L, 2 * L - 1)), axes=1)
    fm = jnp.einsum("lmi, lm->im", legendre_kernel, flm_shift, optimize=True)
    return float(2 * L - 1) * jnp.fft.ifft(fm)
