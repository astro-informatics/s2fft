Differentiable and accelerated spherical transforms
===================================================

``S2FFT`` is a JAX package for computing Fourier transforms on the sphere and rotation 
group.  It leverages autodiff to provide differentiable transforms, which are also 
deployable on modern hardware accelerators (e.g. GPUs and TPUs).

More specifically, ``S2FFT`` provides support for spin spherical harmonic and Wigner
transforms (for both real and complex signals), with support for adjoint transformations
where needed, and comes with different optimisations (precompute or not) that one
may select depending on available resources and desired angular resolution :math:`L`.

Algorithms |:zap:|
-------------------

``S2FFT`` leverages new algorithmic structures that can he highly parallelised and
distributed, and so map very well onto the architecture of hardware accelerators (i.e.
GPUs and TPUs).  In particular, these algorithms are based on new Wigner-d recursions
that are stable to high angular resolution :math:`L`.  The diagram below illustrates the recursions (for further details see Price & McEwen 2023).

.. image:: ./assets/figures/schematic.png

Sampling |:earth_africa:|
-----------------------------------

The structure of the algorithms implemented in ``S2FFT`` can support any isolattitude sampling scheme.  A number of sampling schemes are currently supported.

The equiangular sampling schemes of `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_ and `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_ are supported, which exhibit associated sampling theorems and so harmonic transforms can be computed to machine precision.  Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required. 

The popular `HEALPix <https://healpix.jpl.nasa.gov>`_ sampling scheme (`Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_) is also supported.  The HEALPix sampling does not exhibit a sampling theorem and so the corresponding harmonic transforms do not achieve machine precision but exhibit some error.  However, the HEALPix sampling provides pixels of equal areas, which has many practical advantages.
    
.. image:: ./assets/figures/spherical_sampling.png
   :width: 700
   :align: center

Contributors |:hammer:|
------------------------
``S2FFT`` has been developed at UCL, predominantly by `Matt Price
<https://cosmomatt.github.io/>`_ and `Jason McEwen <http://www.jasonmcewen.org/>`_, with
support from UCL's Advanced Research Computing (ARC) Centre.  The software was
funded in part by a UCL-ARC Open Source Software Sustainability grant. 

We encourage contributions from any interested developers. A simple first addition could be adding 
support for more spherical sampling patterns!

Attribution |:books:|
------------------

Should this code be used in any way, we kindly request that the following
article is referenced. A BibTeX entry for this reference may look like:

.. code-block:: 

     @article{price:s2fft, 
        AUTHOR      = "Matthew A. Price and Jason D. McEwen",
        TITLE       = "TBA",
        YEAR        = "2023",
        EPRINT      = "arXiv:0000.00000"        
     }

You might also like to consider citing our related papers on which this code builds:

.. code-block:: 

    @article{mcewen:fssht,
        AUTHOR      = "Jason D. McEwen and Yves Wiaux",
        TITLE       = "A novel sampling theorem on the sphere",
        JOURNAL     = "IEEE Trans. Sig. Proc.",
        YEAR        = "2011",
        VOLUME      = "59",
        NUMBER      = "12",
        PAGES       = "5876--5887",        
        EPRINT      = "arXiv:1110.6298",
        DOI         = "10.1109/TSP.2011.2166394"
    }

.. code-block::
   
    @article{mcewen:so3,
        AUTHOR      = "Jason D. McEwen and Martin B{\"u}ttner and Boris ~Leistedt and Hiranya V. Peiris and Yves Wiaux",
        TITLE       = "A novel sampling theorem on the rotation group",
        JOURNAL     = "IEEE Sig. Proc. Let.",
        YEAR        = "2015",
        VOLUME      = "22",
        NUMBER      = "12",
        PAGES       = "2425--2429",
        EPRINT      = "arXiv:1508.03101",
        DOI         = "10.1109/LSP.2015.2490676"    
    }

License |:memo:|
----------------

We provide this code under an MIT open-source licence with the hope that it will be of use 
to a wider community. 

Copyright 2023 Matthew Price, Jason McEwen and contributors.

``S2FFT`` is free software made available under the MIT License. For details see
the LICENSE file.

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
   :maxdepth: 3
   :caption: Interactive Tutorials
   
   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   api/index

