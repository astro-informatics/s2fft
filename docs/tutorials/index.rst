:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2FFT`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2FFT`` package is structured such that the 2 primary transforms, the Wigner and 
spherical harmonic transforms, can easily be accessed.

Core usage |:rocket:|
-----------------
To import and use ``S2FFT``  is as simple follows: 

+-------------------------------------------------------+------------------------------------------------------------+
|For a signal on the sphere                             |For a signal on the rotation group                          |
|                                                       |                                                            |
|.. code-block:: Python                                 |.. code-block:: Python                                      |
|                                                       |                                                            |
|   # Compute harmonic coefficients                     |   # Compute Wigner coefficients                            |
|   flm = s2fft.forward_jax(f, L)                       |   flmn = s2fft.wigner.forward_jax(f, L, N)                 |
|                                                       |                                                            |
|   # Map back to pixel-space signal                    |   # Map back to pixel-space signal                         |
|   f = s2fft.inverse_jax(flm, L)                       |   f = s2fft.wigner.inverse_jax(flmn, L, N)                 |
+-------------------------------------------------------+------------------------------------------------------------+

C/C++ backend usage |:bulb:|
-----------------
``S2FFT`` also provides JAX support for existing C/C++ packages, specifically `HEALPix <https://healpix.jpl.nasa.gov>`_
and `SSHT <https://github.com/astro-informatics/ssht>`_.  This works 
by wrapping python bindings with custom JAX frontends. Note that currently this C/C++ to JAX interoperability is currently 
limited to CPU, however for many applications this is desirable due to memory constraints.

For example, one may call these alternate backends for the spherical harmonic transform by:

.. code-block:: python

   # Forward SSHT spherical harmonic transform
   flm = s2fft.forward(f, L, sampling=["mw"], method="jax_ssht")  

   # Forward HEALPix spherical harmonic transform
   flm = s2fft.forward(f, L, nside=nside, sampling="healpix", method="jax_healpy")  

All of these JAX frontends supports out of the box reverse mode automatic differentiation, 
and under the hood is simply linking to the C/C++ packages you are familiar with. In this 
way ``S2FFT`` enhances existing packages with gradient functionality for modern signal processing 
applications!


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   spherical_harmonic/spherical_harmonic_transform.nblink
   wigner/wigner_transform.nblink
   rotation/rotation.nblink
   torch_frontend/torch_frontend.nblink
   JAX_SSHT/JAX_SSHT_frontend.nblink
   JAX_HEALPix/JAX_HEALPix_frontend.nblink
