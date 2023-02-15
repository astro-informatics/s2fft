.. image:: https://img.shields.io/badge/GitHub-S2FFT-blue.svg?style=flat
    :target: https://github.com/astro-informatics/s2fft
.. image:: https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml
.. image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/s2fft
.. image:: https://codecov.io/gh/astro-informatics/s2fft/branch/main/graph/badge.svg?token=7QYAFAAWLE
    :target: https://codecov.io/gh/astro-informatics/s2fft
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. image:: http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat
    :target: https://arxiv.org/abs/xxxx.xxxxx
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

|logo| Primary project directory
=================================================================================================================

.. |logo| raw:: html

   <img src="../docs/assets/sax_logo.png" align="left" height="85" width="98">

This directory contains further sub-directories that handle a variety of different 
*modus operandi* for ``S2FFT``. 

**In base transforms:** we have basic python implementations 
of classic spherical harmonic and wigner transforms, which should demonstrate how 
accelerated algorithms emerge from inefficient nested python loops. 

**In precompute transforms:** we provide both numpy and JAX implementations of the spherical 
harmonic and Wigner transforms, where we split the long/latitudinal components by separation 
of variables, resulting in a simple Fast Fourier Transform in longitude with complexity 
O(L^2Log(2L)) and a more complex O(L^3) operation in latitude. For the precompute method 
we precompute all kernels associated with this latitudinal transform, which then reduces 
to an extremely fast ``jnp.einsum`` but becomes extremely memory inefficient O(L^3).

**In transforms:** we provide the core ``S2FFT`` functions which evaluate the latitudinal step 
recursively on-the-fly with at most O(L^2) memory overhead. These transforms come both 
in standard numpy and JAX varients, but of course we highly recommend JAX.


.. image:: ../docs/assets/figures/software_overview.png