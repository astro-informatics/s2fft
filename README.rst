.. image:: https://img.shields.io/badge/GitHub-s2fft-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/s2fft
.. image:: https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml
.. image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/s2fft
.. image:: https://codecov.io/gh/astro-informatics/s2fft/branch/main/graph/badge.svg?token=7QYAFAAWLE
    :target: https://codecov.io/gh/astro-informatics/s2fft
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat
    :target: https://arxiv.org/abs/xxxx.xxxxx

|logo| S2FFT: Differentiable and accelerated spin-spherical harmonic transforms with JAX
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/placeholder_logo.png" align="center" height="80" width="80">

This is a very loose framework for people to start python projects from. To get up and running, go through the code carefully replacing ``Project-name`` with your 
desired project name (check all documents!), don't forget to change the directory s2fft as well! You will also need to update the links in all badges!

Installation
============

The Python dependencies for the `s2fft` package are listed in the file ``requirements/requirements-core.txt`` and can be installed 
into the active Python environment using `pip` by running

.. code-block:: bash 

    pip install -r requirements/requirements-core.txt
    
from the root directory of the repository.
    
To install the `s2fft` package in editable mode in the current Python environment run

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

flm = forward_transform(f, L, sampling, reality, implementation)
f = inverse_transform(flm, sampling, reality, implementation, nside=None)

sampling = {"mw", "mwss", "healpix"}; default = mw
reality = {"real", "complex"}; default = complex
implementation = {"loopy", "vectorized", "jax"}; default = jax
nside default = None



Auto-formatting code
====================
To keep the code readable and organised you should (strongly) consider using the ``black`` package. Whenever you are finished updating a file, just run 

.. code-block:: bash

    black <file_to_tidy.py>

or alternatively format everything by running

.. code-block:: bash

    black s2fft/*

This is important as the CI enforces black formatting (this can be disabled by removing the --black flag in pytest) so your unit tests will fail if you don't do this!

CodeCov
============
To set up code coverage you will need to enter this  

.. code-block:: bash

    https://codecov.io/gh/{account-name}/{desired-repo} 

into any browser, then go to settings and activate the repository. You will then need to find the ``repository upload token`` which 
should be added to the github actions script (roughly line 29)

.. code-block::

    codecov --token <add your token here>

Next time CI runs on main branch it will automatically update codecov. Now go back to codecov, copy the badge and put it in the readme, .pipreadme, and 
the root index of the documentation!

PyPi
=====
To deploy the code on PyPi first test the deployment on PyPi's mirror site by, first making an account on https://test.pypi.org and then running 

.. code-block:: bash 

    python setup.py bdist_wheel --universal
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    pip install -i https://test.pypi.org/simple/ s2fft

From the root directory. Keep in mind that installing from the mirror site won't automatically find dependencies, so if you have an error because the pacakge can't find numpy that's probably why, and may not be an issue on the main PyPi site. To deploy the main PyPi site simply remove the --repostiry-url name, note that you can add multiple wheels to dist/*, to provide a package which may be pip installed for multiple python version, and on multiple machine architectures.

Attribution
===========
A BibTeX entry for <project-name> is:

.. code-block:: 

     @article{<project-name>, 
        author = {Author~List},
         title = {"A totally amazing name"},
       journal = {ArXiv},
        eprint = {arXiv:0000.00000},
          year = {what year is it?!}
     }

License
=======

``<project-name>`` is released under the MIT license (see `LICENSE.txt <https://github.com/astro-informatics/code_template/blob/main/LICENCE.txt>`_).

.. code-block::

     S2fft
     Copyright (C) 2022 Author names & contributors

     This program is released under the MIT license (see LICENCE.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
