"""
CUDA-Accelerated HEALPix Transforms with S2FFT
==============================================

This notebook demonstrates how to use CUDA-accelerated HEALPix spherical harmonic transforms in S2FFT.

The CUDA implementation provides:

* Fast JIT compilation using pre-compiled cuFFT and custom CUDA kernels
* Performance comparable to pure JAX on GPU
* Full compatibility with JAX transformations (vmap, grad, jacfwd, jacrev)
"""

x = 5
