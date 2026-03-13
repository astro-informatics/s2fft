Tutorials
=========

This section contains :ref:`a series of tutorial notebooks <tutorial-notebooks-label>` which go through some of the
key features of the ``S2FFT`` package.

At a high-level the ``S2FFT`` package is structured such that the two primary transforms,
the Wigner and spherical harmonic transforms, can easily be accessed.

Core usage |:rocket:|
---------------------

To import and use ``S2FFT`` is as simple follows: 

+-------------------------------------------------------+------------------------------------------------------------+
|For a signal on the sphere                             |For a signal on the rotation group                          |
|                                                       |                                                            |
|.. code-block:: Python                                 |.. code-block:: Python                                      |
|                                                       |                                                            |
|   import s2fft                                        |   import s2fft                                             |
|                                                       |                                                            |
|   # Specify sampled signal and harmonic bandlimit     |   # Define sampled signal, harmonic & azimuthal bandlimits |
|   f = ...                                             |   f = ...                                                  |
|   L = ...                                             |   L, N = ...                                               |
|                                                       |                                                            |
|   # Compute harmonic coefficients                     |   # Compute Wigner coefficients                            |
|   flm = s2fft.forward(f, L, method="jax")             |   flmn = s2fft.wigner.forward(f, L, N, method="jax")       |
|                                                       |                                                            |
|   # Map back to pixel-space signal                    |   # Map back to pixel-space signal                         |
|   f = s2fft.inverse(flm, L, method="jax")             |   f = s2fft.wigner.inverse(flmn, L, N, method="jax")       |
+-------------------------------------------------------+------------------------------------------------------------+

.. _tutorial-notebooks-label:

Tutorial notebooks
------------------

Below are a few short tutorials that cover how to use specific features of ``S2FFT``.

We also have a notebook demonstrating how to use CUDA-accelerated HEALPix spherical harmonic transforms in ``S2FFT``, which `is accessible in notebook format here <https://github.com/astro-informatics/s2fft/blob/main/notebooks/JAX_CUDA_HEALPix.ipynb>`_, or alternatively can be `opened in Google Colab <https://colab.research.google.com/github/astro-informatics/s2fft/blob/main/notebooks/JAX_CUDA_HEALPix.ipynb>`_.
