Sampling Schemes
================

The structure of the algorithms implemented in ``S2FFT`` can support any isolatitude sampling scheme.
A number of sampling schemes are currently supported.

The equiangular sampling schemes of `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_, `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_, and `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_ are supported, which exhibit associated sampling theorems and so
harmonic transforms can be computed to machine precision.
Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required.

The popular `HEALPix <https://healpix.jpl.nasa.gov>`_ sampling scheme (`Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_) is also supported.
The HEALPix sampling does not exhibit a sampling theorem and so the corresponding harmonic transforms do not achieve machine precision but exhibit some error.
However, the HEALPix sampling provides pixels of equal areas, which has many practical advantages.

.. image:: https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/spherical_sampling.png
    :width: 700
    :alt: Visualization of spherical sampling schemes
    :align: center

.. list-table:: At-a-glance comparison of sampling schemes
    :header-rows: 1
    :align: center

    * - Scheme
      - API string
      - Number of samples
      - (I)FFT complexity
      - Stable to band-limit
      - Has sampling theorem
    * - McEwen & Wiaux
      - ``"mw"``
      - Order $2L^2$
      - :math:`\mathcal{O}(L^3)`
      - :math:`L \approx 4096`
      - Yes
    * - McEwen & Wiaux SS
      - ``"mwss"``
      - Order
      - :math:`\mathcal{O}()`
      - :math:`L \approx`
      - Yes
    * - Driscoll & Healy
      - ``"dh"``
      - Order $4L^2$
      - :math:`\mathcal{O}(L^2(\log L)^2)`
      - :math:`1024 \leq L \leq 2048`
      - Yes
    * - Gauss-Legendre
      - ``"gl"``
      - Order $2L^2$
      - :math:`\mathcal{O}(L^3)`
      - :math:`1024 \leq L \leq 2048`
      - Yes
    * - HEALPix
      - ``"healpix"``
      - Order
      - :math:`\mathcal{O}()`
      - :math:`L \approx`
      - No

Specifying Sampling Schemes with the ``S2FFT`` API
--------------------------------------------------

Most public ``S2FFT`` functions accept a ``sampling`` optional argument, which will be interpreted as the sampling scheme to be used in that call to the function.
Sampling schemes are specified by providing the corresponding character string to the ``sampling`` argument (listed in the "API string" column of the table above).

Not all functions support all available sampling schemes; the function docstring will clarify which schemes are accepted by the function.
