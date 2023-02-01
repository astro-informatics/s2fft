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

|logo| Accelerated and differentiable spherical transforms with JAX
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/sax_logo.png" align="left" height="85" width="98">

A JAX package for Generalised Fast Fourier Transforms (GFFTs) on the sphere and rotation 
group, which is differentiable, and deployable on modern hardware accelerators (GPU & TPUs).

Overview
---------
``S2FFT`` is a software package which provides support for Generalised Fast Fourier Transforms 
on the sphere and the rotation group. Leveraging the highly engineered Price-McEwen 
Wigner-d recursions our transforms exhibit a highly parallelisable algorithmic structure, 
and are numerically stable beyond :math:`L > 20,000`. Moreover, these JAX transforms are 
not only automatically differentiable and deployable on accelerators (GPU & TPUs), but they 
are also sampling agnostic; all that is required are latitudinal samples on the sphere 
and appropriate quadrature weights. As such we support 
`McEwen-Wiaux <https://arxiv.org/abs/1110.6298>`_,  and `HEALPix <https://healpix.jpl.nasa.gov>`_ 
in addition to various other discretisations of the sphere.

.. note::
   By construction ``S2FFT`` is straightforward to install, provides support 
   for spin-spherical harmonic and Wigner transforms (over both real and complex signals), 
   with straightforward extensions to adjoint transformations where needed, and comes 
   with various different optimisations depending on available compute and/or memory.


Installation
------------

The Python dependencies for the ``S2FFT`` package are listed in the file ``requirements/requirements-core.txt`` and can be installed 
into the active Python environment using `pip` by running

.. code-block:: bash 

    pip install -r requirements/requirements-core.txt
    
from the root directory of the repository.
    
To install the ``S2FFT`` package in editable mode in the current Python environment run

.. code-block:: bash
    
    pip install -e .
    
from the root directory of the repository.


Tests
-----

To run the tests the additional dependencies in the file ``requirements/requirements-test.txt`` will also need to be installed in the active environment.

The tests can then be run directly

.. code-block:: bash
    
    pytest tests
    
from the root directory of the repository, or if ``tox`` is installed then it can be used to create a temporary virtual environment with the package dependencies and run the tests in that environment using

.. code-block:: bash
    
    tox -e py38
    

Documentation
--------------

To build the documentation the additional dependencies in the file ``requirements/requirements-docs.txt`` will also need to be installed in the active environment. 

Pandoc also needs to be installed to allow building the tutorial notebook documentation - this can be done using ``conda`` by running

.. code-block:: bash
    
    conda install -c conda-forge pandoc
    
Note that this installs the Pandoc Haskell library and command-line tool rather than [the `pandoc` Python package on PyPI](https://pypi.org/project/pandoc/) which wraps this library.

The HTML documentation can then be built by running

.. code-block:: bash
    
    sphinx-build -M html docs docs/_build -Q
    
from the root directory of the repository, or if ``tox`` is installed then it can be used to build the HTML documentation by running

.. code-block:: bash
    
    tox -e docs
    
from the root directory of the repository.


Interface
---------

Temporary notes on interface to be updated.

.. code-block:: python

    flm = forward_transform(f, L, sampling, reality, implementation)
    f = inverse_transform(flm, sampling, reality, implementation, nside=None)

    sampling = {"mw", "mwss", "healpix"}; default = mw
    reality = {"real", "complex"}; default = complex
    implementation = {"loopy", "vectorized", "jax"}; default = jax
    nside default = None


Attribution
-----------
A BibTeX entry for ``S2FFT`` is:

.. code-block:: 

     @article{S2FFT, 
        author = {Author~List},
         title = {"A totally amazing name"},
       journal = {ArXiv},
        eprint = {arXiv:0000.00000},
          year = {what year is it?!}
     }

License
-------

``S2FFT`` is released under the MIT license (see 
`LICENSE.txt <https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt>`_).

.. code-block::

     S2FFT
     Copyright (C) 2023 Author names & contributors

     This program is released under the MIT license (see `LICENSE.txt`).
