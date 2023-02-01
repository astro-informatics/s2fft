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
``S2FFT`` is a core module from the ``SAX`` package, which includes extensions of these transforms 
to accelerated, differentiable, and directional spin-wavelets, ``S2WAV`` on both the 2-sphere 
`McEwen et al 2015 <https://arxiv.org/abs/1509.06749>`_ and 3-ball 
`Price & McEwen 2021 <https://arxiv.org/abs/2105.05518>`_.

Overview
---------
``S2FFT`` is a software package which provides support for Generalised Fast Fourier Transforms 
on the sphere and the rotation group. Leveraging the highly engineered Price-McEwen 
Wigner-d recursions our transforms exhibit a highly parallelisable algorithmic structure, 
and are theoretically indefinitely numerically stable; certainly far beyond :math:`L > 20,000` although 
64bit floating point errors will begin to accumulate eventually. Moreover, these JAX transforms 
are not only automatically differentiable and deployable on accelerators (GPU & TPUs), but they 
are also sampling agnostic; all that is required are latitudinal samples on the sphere and 
appropriate quadrature weights. As such we support `McEwen-Wiaux <https://arxiv.org/abs/1110.6298>`_, 
`HEALPix <https://healpix.jpl.nasa.gov>`_, and `Driscoll-Healy <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_ 
in addition to various other discretisations of the sphere.

    **NOTE:**
    By construction ``S2FFT`` is straightforward to install, provides support 
    for spin-spherical harmonic and Wigner transforms (over both real and complex signals), 
    with straightforward extensions to adjoint transformations where needed, and comes 
    with various different optimisations depending on available compute and/or memory.


Installation
------------

Source 

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
    

Contributors
--------------
The development of ``S2FFT`` is one aspect of the ``SAX`` collaborative project between 
the Mullard Space Science Laboratory (MSSL) and Advanced Research Computing (ARC), which aims 
to develop accelerated and differentiable spherical transforms to enable ongoing research 
into next-generation informatics techniques on the 2-sphere and rotation group.
Both academic groups are based at University College London (UCL) and this software was, in part, 
funded by a UCL-ARC Open Source Software Sustainability grant. The development group includes: 
`Matthew A. Price <https://cosmomatt.github.io/>`_ (MSSL, PI), 
`Jason D. McEwen <http://www.jasonmcewen.org/>`_ (MSSL, Alan Turing Institute), 
`Matthew Graham <https://matt-graham.github.io>`_ (ARC),
`Sofía Miñano <https://www.linkedin.com/in/sofiaminano/?originalSubdomain=uk>`_ (ARC),
`Devaraj Gopinathan <https://www.linkedin.com/in/devaraj-g/?originalSubdomain=uk>`_ (ARC), 
pictured below left to right.

.. image:: ./docs/assets/authors/price.jpeg
   :width: 155
   :target: https://cosmomatt.github.io/


.. image:: ./docs/assets/authors/mcewen.jpeg
   :width: 155
   :target: http://www.jasonmcewen.org/


.. image:: ./docs/assets/authors/graham.jpeg
   :width: 155
   :target: https://matt-graham.github.io


.. image:: ./docs/assets/authors/minano.jpeg
   :width: 155
   :target: https://www.linkedin.com/in/sofiaminano/?originalSubdomain=uk


.. image:: ./docs/assets/authors/gopinathan.jpeg
   :width: 155
   :target: https://www.linkedin.com/in/devaraj-g/?originalSubdomain=uk



Attribution
--------------
We provide this code under an MIT open-source licence with the hope that it will be of use 
to a wider community. Should this code be used in any way, we kindly request that the follow 
article is correctly referenced. A BibTeX entry for this reference may look like:

.. code-block:: 

     @article{price:2023:sax, 
        author = {Price, Matthew A and McEwen, Jason D and Graham, Matthew and Miñano-González, Sofía and Gopinathan, Devaraj},
         title = {"Name pending"},
       journal = {ArXiv},
        eprint = {arXiv:0000.00000},
          year = {2023}
     }
License
-------

``S2FFT`` is released under the MIT license (see 
`LICENSE.txt <https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt>`_).

.. code-block::

     S2FFT
     Copyright (C) 2023 Author names & contributors

     This program is released under the MIT license (see `LICENSE.txt`).
