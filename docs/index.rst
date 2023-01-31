Accelerated and differentiable spherical harmonic and Wigner transforms with JAX
=================================================================================================================

``S2FFT`` is a software package which provides support for Generalised Fast Fourier Transforms 
on the sphere and the rotation group. Leveraging the highly engineered Price-McEwen 
Wigner-d recursions our transforms exhibit a highly parallelisable algorithmic structure, 
and are numerically stable beyond :math:`L > 20,000`. Moreover, these JAX transforms are 
not only automatically differentiable and deployable on accelerators (GPU & TPUs), but they 
are also sampling agnostic; all that is required are latitudinal samples on the sphere 
and appropriate quadrature weights. As such we support `McEwen-Wiaux <https://arxiv.org/abs/1110.6298>`_, 
and `HEALPix <https://healpix.jpl.nasa.gov>`_ in addition to various other discretisations of the sphere.

.. note::
   By construction ``S2FFT`` is straightforward to install, provides support 
   for spin-spherical harmonic and Wigner transforms (over both real and complex signals), 
   with straightforward extensions to adjoint transformations where needed, and comes 
   with various different optimisations depending on available compute and/or memory.


Contributors
--------------
The development of ``S2FFT`` is one aspect of the ``SAX`` collaborative project between 
the Mullard Space Science Laboratory (MSSL) and Advanced Research Computing (ARC), which aims 
to develop accelerated and differentiable spherical transforms to enable ongoing research 
into next-generation informatics techniques on :math:`\mathbb{S}^2` and SO(3).
Both academic groups are based at University College London (UCL) and this software was, in part, 
funded by a UCL-ARC Open Source Software Sustainability grant. The development group includes: 
`Matthew A. Price <https://cosmomatt.github.io/>`_ (MSSL, PI), 
`Jason D. McEwen <http://www.jasonmcewen.org/>`_ (MSSL, Alan Turing Institute), 
`Matthew Graham <https://matt-graham.github.io>`_ (ARC),
`Sofía Miñano <https://www.linkedin.com/in/sofiaminano/?originalSubdomain=uk>`_ (ARC),
`Devaraj Gopinathan <https://www.linkedin.com/in/devaraj-g/?originalSubdomain=uk>`_ (ARC), 
pictured below left to right.

.. image:: assets/authors/price.jpeg
   :width: 155
   :target: https://cosmomatt.github.io/


.. image:: assets/authors/mcewen.jpeg
   :width: 155
   :target: http://www.jasonmcewen.org/


.. image:: assets/authors/graham.jpeg
   :width: 155
   :target: https://matt-graham.github.io


.. image:: assets/authors/minano.jpeg
   :width: 155
   :target: https://www.linkedin.com/in/sofiaminano/?originalSubdomain=uk


.. image:: assets/authors/gopinathan.jpeg
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
--------------

``S2FFT`` is released under the MIT license (see `LICENSE.txt <https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt>`_).

.. code-block::

     S2fft
     Copyright (C) 2023 Matthew A Price & contributors

     This program is released under the MIT license.

.. bibliography:: 
    :notcited:
    :list: bullet

.. * :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Benchmarking

   benchmarking/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   background/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Interactive Tutorials
   
   tutorials/example_notebook.nblink

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   api/index

