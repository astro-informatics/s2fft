Sampling Schemes
----------------

The structure of the algorithms implemented in `S2FFT` can support any isolatitude sampling scheme.
A number of sampling schemes are currently supported.

The equiangular sampling schemes of `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_, `Driscoll & Healy
(1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_, and `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_ are supported, which exhibit associated sampling theorems and so
harmonic transforms can be computed to machine precision.
Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required.

The popular `HEALPix <https://healpix.jpl.nasa.gov>`_ sampling scheme (`Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_) is also supported.
The HEALPix sampling does not exhibit a sampling theorem and so the corresponding harmonic transforms do not achieve machine precision but exhibit some error.
However, the HEALPix sampling provides pixels of equal areas, which has many practical advantages.

.. image:: https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/spherical_sampling.png
    :width: 700
    :alt: Visualization of spherical sampling schemes
    :align: center
