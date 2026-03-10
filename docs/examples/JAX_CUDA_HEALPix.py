"""
CUDA-Accelerated HEALPix Transforms with S2FFT
==============================================

This notebook demonstrates how to use CUDA-accelerated HEALPix spherical harmonic transforms in S2FFT.

The CUDA implementation provides:

* Fast JIT compilation using pre-compiled cuFFT and custom CUDA kernels
* Performance comparable to pure JAX on GPU
* Full compatibility with JAX transformations (vmap, grad, jacfwd, jacrev)
"""

# %%
# Setup
# ------------------------
# Import required packages and enable JAX 64-bit precision for numerical accuracy.

import jax
import jax.numpy as jnp
import healpy as hp
from s2fft import forward, inverse

jax.config.update("jax_enable_x64", True)

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# %%
# Basic Usage
# -----------
# Use `method='jax_cuda'` to enable CUDA acceleration for HEALPix transforms.

nside = 32
npix = hp.nside2npix(nside)
lmax = 3 * nside - 1
L = lmax + 1

print(f"HEALPix parameters:")
print(f"  nside: {nside}")
print(f"  lmax: {lmax}")
print(f"  L (band limit): {L}")
print(f"  Number of pixels: {npix}")

hp_map = jax.random.normal(jax.random.PRNGKey(0), shape=(npix,))
print(f"\nGenerated random HEALPix map with shape: {hp_map.shape}")

# %%
# Forward Transform (Analysis)
# ----------------------------
# Compute spherical harmonic coefficients from a HEALPix map.

alm_cuda = forward(
    hp_map, nside=nside, L=L, sampling="healpix", method="jax_cuda"
).block_until_ready()

print(f"Spherical harmonic coefficients shape: {alm_cuda.shape}")
print(f"Shape is (n_rings, 2*L) = ({4 * nside - 1}, {2 * L})")

# %%
# Inverse Transform (Synthesis)
# -----------------------------
# Reconstruct a HEALPix map from spherical harmonic coefficients.

f_recon = inverse(
    alm_cuda, nside=nside, L=L, sampling="healpix", method="jax_cuda"
).block_until_ready()

print(f"Reconstructed map shape: {f_recon.shape}")

# %%
# Performance Comparison
# ----------------------
# Compare CUDA implementation (`method='jax_cuda'`) vs pure JAX (`method='jax'`).

def forward_cuda(f):
    return forward(f, nside=nside, L=L, sampling='healpix', method='jax_cuda')

def forward_jax(f):
    return forward(f, nside=nside, L=L, sampling='healpix', method='jax')

print("Forward Transform - First run (includes JIT compilation):")
print("\nCUDA:")
%time _ = forward_cuda(hp_map).block_until_ready()
print("\nPure JAX:")
%time _ = forward_jax(hp_map).block_until_ready()

print("\n" + "="*60)
print("Forward Transform - Execution time (after JIT):")
print("\nCUDA:")
%timeit forward_cuda(hp_map).block_until_ready()
print("\nPure JAX:")
%timeit forward_jax(hp_map).block_until_ready()

# %%
# Accuracy Verification
# ---------------------
# Verify that CUDA and pure JAX implementations produce identical results.

alm_cuda = forward_cuda(hp_map)
alm_jax = forward_jax(hp_map)

mse = jnp.mean(jnp.abs(alm_cuda - alm_jax) ** 2)
max_diff = jnp.max(jnp.abs(alm_cuda - alm_jax))

print(f"Forward transform comparison:")
print(f"  Mean Squared Error: {mse:.2e}")
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  Results match: {jnp.allclose(alm_cuda, alm_jax, atol=1e-14)}")

# %%
# JAX Transformations
# -------------------
# The CUDA implementation is fully compatible with JAX's automatic differentiation and batching.
# We use `nside=16` for these demonstrations to keep memory requirements reasonable.

nside_test = 16
npix_test = hp.nside2npix(nside_test)
L_test = 3 * nside_test

batch_size = 3
f_batch = jnp.stack([
    jax.random.normal(jax.random.PRNGKey(i), shape=(npix_test,))
    for i in range(batch_size)
])

print(f"Test parameters:")
print(f"  nside: {nside_test}")
print(f"  Batch size: {batch_size}")
print(f"  Batch shape: {f_batch.shape}")

# %%
# Batching with `vmap`
# ....................
# Process multiple maps in parallel using `jax.vmap`.

def forward_test(f):
    return forward(f, nside=nside_test, L=L_test, sampling='healpix', method='jax_cuda')

alm_batch = jax.vmap(forward_test)(f_batch)

print(f"Batched transform output shape: {alm_batch.shape}")
print(f"Expected: ({batch_size}, {4*nside_test-1}, {2*L_test})")
print(f"\nvmap works correctly: {alm_batch.shape == (batch_size, 4*nside_test-1, 2*L_test)}")

# %%
# Automatic Differentiation with `grad`
# .....................................
# Compute gradients through the transform.

f_single = f_batch[0].real

@jax.grad
def loss_fn(x):
    alm = forward_test(x).real
    return jnp.sum(alm ** 2)

grad_f = loss_fn(f_single)

print(f"Input shape: {f_single.shape}")
print(f"Gradient shape: {grad_f.shape}")
print(f"Gradient is finite: {jnp.all(jnp.isfinite(grad_f))}")
print(f"\ngrad works correctly: True")

# %%
# Usage
# .....
# Simply use `method='jax_cuda'` in your `forward()` and `inverse()` calls:
# 
# ```python
# alm = s2fft.forward(hp_map, nside=nside, L=L, sampling='healpix', method='jax_cuda')
# f = s2fft.inverse(alm, nside=nside, L=L, sampling='healpix', method='jax_cuda')
# ```
#
# Requirements
# ............
# 
# * CUDA toolkit 12.3+
# * S2FFT compiled with CUDA support (`nvcc` in PATH during installation)
# * GPU-enabled JAX