:html_theme.sidebar_secondary.remove:

*****************************
Tutorials
*****************************

This section contains a series of tutorial notebooks which go through some of the
key features of the ``S2FFT`` package.

At a high-level the ``S2FFT`` package is structured such that the two primary transforms,
the Wigner and spherical harmonic transforms, can easily be accessed.

Core usage |:rocket:|
-----------------------------

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

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Tutorials

   spherical_harmonic/spherical_harmonic_transform.nblink
   wigner/wigner_transform.nblink
   rotation/rotation.nblink
   torch_frontend/torch_frontend.nblink
   JAX_SSHT/JAX_SSHT_frontend.nblink
   JAX_HEALPix/JAX_HEALPix_frontend.nblink
