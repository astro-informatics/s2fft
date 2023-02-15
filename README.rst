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

|logo| Differentiable and accelerated spherical transforms with JAX
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/sax_logo.png" align="left" height="85" width="98">

``S2FFT`` is a JAX package for computing Fourier transforms on the sphere and rotation 
group.  It leverages autodiff to provide differentiable transforms, which are also 
deployable on modern hardware accelerators (e.g. GPUs and TPUs). 

More specifically, ``S2FFT`` provides support for spin spherical harmonic and Wigner
transforms (for both real and complex signals), with support for adjoint transformations
where needed, and comes with different optimisations (precompute or not) that one
may select depending on available resources and desired angular resolution :math:`L`.

Algorithms :zap:
----------------

``S2FFT`` leverages new algorithmic structures that can he highly parallelised and
distributed, and so map very well onto the architecture of hardware accelerators (i.e.
GPUs and TPUs).  In particular, these algorithms are based on new Wigner-d recursions
that are stable to high angular resolution :math:`L`.  The diagram below illustrates the recursions (for further details see Price & McEwen 2023).

.. image:: ./docs/assets/figures/schematic.png

Sampling :globe_with_meridians:
-----------------------------------

The structure of the algorithms implemented in ``S2FFT`` can support any isolattitude sampling scheme.  A number of sampling schemes are currently supported.

The equiangular sampling schemes of `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_ and `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_ are supported, which exhibit associated sampling theorems and so harmonic transforms can be computed to machine precision.  Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required. 

The popular `HEALPix <https://healpix.jpl.nasa.gov>`_ sampling scheme (`Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_) is also supported.  The HEALPix sampling does not exhibit a sampling theorem and so the corresponding harmonic transforms do not achieve machine precision but exhibit some error.  However, the HEALPix sampling provides pixels of equal areas, which has many practical advantages.
    
.. image:: ./docs/assets/figures/spherical_sampling.png
    :width: 400

Installation :computer:
------------------------
The Python dependencies for the ``S2FFT`` package are listed in the file 
``requirements/requirements-core.txt`` and will be automatically installed into the 
active python environment by `pip` when running

.. code-block:: bash 

    pip install .        
    
from the root directory of the repository. Unit tests can then be executed to ensure the 
installation was successful by running 

.. code-block:: bash 

    pytest tests/         # for pytest
    tox -e py38           # for tox 

In the very near future one will be able to install ``S2FFT`` directly from `PyPi` by ``pip install s2fft`` but this is not yet supported.

Usage :rocket:
--------------
To import and apply the ``S2FFT`` apis is as simple as doing the following 

+-------------------------------------------------------+------------------------------------------------------------+
|for a spin signal on the sphere                        |for a signal on the rotation group                          |
|                                                       |                                                            |
|.. code-block:: Python                                 |.. code-block:: Python                                      |
|                                                       |                                                            |
|   # Compute harmonic coefficients                     |   # Compute Wigner coefficients                            |
|   flm = s2fft.forward_jax(f, L, spin)                 |   flmn = s2fft.wigner.forward_jax(f, L, N)                 |
|                                                       |                                                            |
|   # Map back to pixel-space signal                    |   # Map back to pixel-space signal                         |
|   f = s2fft.inverse_jax(flm, L, spin)                 |   f = s2fft.wigner.inverse_jax(flmn, L, N)                 |
+-------------------------------------------------------+------------------------------------------------------------+

For repeated application of the transforms, however, it is beneficial to precompute some small constant matrices that 
are used within every transform. To do this simply run 

.. code-block:: Python

    import s2fft 

    precomputes_sphere = s2fft.generate_precomputes_jax(L, spin)
    precomputes_wigner = s2fft.generate_precomputes_wigner_jax(L, N)

which are then passed as `precomps` to the transforms. For signals bandlimited below 
L~1024 we also include a (memory inefficient, but very fast) full precompute mode where 
the wigner-d kernels are precomputed *a priori* and replace the latitudinal (expensive) 
step with a simple JAX einsum.


Benchmarking :hourglass_flowing_sand:
-------------------------------------
We benchmarked the spin-spherical harmonic and Wigner transforms provided by this package 
against their contemporaries, in a variety of settings. We consider both complex signals 
(solid lines) and real signals (dashed lines) wherein hermitian symmetry halves memory 
overhead and wall-time. We further consider single-program multiple-data (SPMD) deployment 
of ``S2FFT``, wherein the compute is distributed across multiple GPUs. Below are 
the results for McEwen-Wiaux sampling for the recursion (left) and precompute (right) 
based spin-spherical harmonic transforms.

+------+-----------+-----------+----------+-----------+----------+----------+---------+
|      |       Recursive Algorithm        |       Precompute Algorithm                |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| L    | Wall-Time | Speed-up  | Error    | Wall-Time | Speed-up | Error    | Memory  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 64   | 3.6 ms    | 0.88      | 1.81E-15 | 52.4 μs   | 60.5     | 1.67E-15 | 4.2 MB  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 128  | 7.26 ms   | 1.80      | 3.32E-15 | 162 μs    | 80.5     | 3.64E-15 | 33 MB   |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 256  | 17.3 ms   | 6.32      | 6.66E-15 | 669 μs    | 163      | 6.74E-15 | 268 MB  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 512  | 58.3 ms   | 11.4      | 1.43E-14 | 3.6 ms    | 184      | 1.37E-14 | 2.14 GB |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 1024 | 194 ms    | 32.9      | 2.69E-14 | 32.6 ms   | 195      | 2.47E-14 | 17.1 GB |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 2048 | 1.44 s    | 49.7      | 5.17E-14 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 4096 | 8.48 s    | 133.9     | 1.06E-13 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 8192 | 82 s      | 110.8     | 2.14E-13 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+

These benchmarks are entirely independent from spin number, however some packages have 
highly optimised (so called 'semi-naive') transforms for scalar spherical harmonic transforms 
which may be extended to spin-signals, and therefore Wigner transforms, by repeated applications 
of spin-raising and spin-lowering operators. This process increases their computation time 
linearly in spin-number, and therefore benchmarking in these settings are highly situation 
dependant. In the scalar case (spin = 0), and for a single GPU, we recover very similar 
compute times, whilst for larger spins the improvement roughly grows to that displayed 
above. 

Contributors :hammer:
------------------------
The development of ``S2FFT`` is one aspect of the ``SAX`` collaborative project between 
the Mullard Space Science Laboratory (MSSL) and Advanced Research Computing (ARC), which aims 
to develop accelerated and differentiable spherical transforms to enable ongoing research 
into next-generation informatics techniques on the 2-sphere and rotation group.
Both academic groups are based at University College London (UCL) and this software was, in part, 
funded by a UCL-ARC Open Source Software Sustainability grant. We strongly encourage 
constributions from any developers that are interested; a simple example would be adding 
support for more spherical sampling patterns!

Attribution :books:
------------------
We provide this code under an MIT open-source licence with the hope that it will be of use 
to a wider community. Should this code be used in any way, we kindly request that the follow 
article is correctly referenced. A BibTeX entry for this reference may look like:

.. code-block:: 

     @article{price:2023:sax, 
        author = {Price, Matthew A and McEwen, Jason D},
         title = {'TBA'},
       journal = {ArXiv},
        eprint = {arXiv:0000.00000},
          year = {2023}
     }
     