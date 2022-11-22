.. image:: https://img.shields.io/badge/GitHub-s2fft-brightgreen.svg?style=flat
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

|logo| S2FFT: Differentiable and accelerated spin-spherical harmonic transforms with JAX
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/placeholder_logo.png" align="center" height="80" width="80">

A description of the overall software package.

Installation
============

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
=====

To run the tests the additional dependencies in the file ``requirements/requirements-test.txt`` will also need to be installed in the active environment.

The tests can then be run directly

.. code-block:: bash
    
    pytest tests
    
from the root directory of the repository, or if ``tox`` is installed then it can be used to create a temporary virtual environment with the package dependencies and run the tests in that environment using

.. code-block:: bash
    
    tox -e py38
    

Documentation
=============

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
=========

Temporary notes on interface to be updated.

.. code-block:: python

    flm = forward_transform(f, L, sampling, reality, implementation)
    f = inverse_transform(flm, sampling, reality, implementation, nside=None)

    sampling = {"mw", "mwss", "healpix"}; default = mw
    reality = {"real", "complex"}; default = complex
    implementation = {"loopy", "vectorized", "jax"}; default = jax
    nside default = None


Attribution
===========
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
=======

``S2FFT`` is released under the MIT license (see `LICENSE.txt <https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt>`_).

.. code-block::

     S2fft
     Copyright (C) 2022 Author names & contributors

     This program is released under the MIT license (see `LICENSE.txt <https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt>`_).
