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

|logo| S2FFT: JAX accelerated spin-spherical harmonic transforms
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/placeholder_logo.png" align="center" height="80" width="80">

This is a very loose framework for people to start python projects from. To get up and running, go through the code carefully replacing ``Project-name`` with your 
desired project name (check all documents!), don't forget to change the directory s2fft as well! You will also need to update the links in all badges!

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

``<project-name>`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/code_template/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/code_template/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     LatentWaves
     Copyright (C) 2022 Author names & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
