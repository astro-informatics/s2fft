Sampling Schemes
================

The structure of the algorithms implemented in ``S2FFT`` can support a number of sampling schemes.

The equiangular sampling schemes of McEwen & Wiaux [#mw]_, Driscoll & Healy [#dh]_, and Gauss-Legendre [#gl]_ are supported, which exhibit associated sampling theorems and so harmonic transforms can be computed to machine precision.

The popular `HEALPix <https://healpix.jpl.nasa.gov>`_ sampling scheme (Gorski et al. [#hp]_ ) is also supported.
HEALPix sampling does not exhibit a sampling theorem and so the corresponding harmonic transforms do not achieve machine precision but exhibit some error.
However, the HEALPix sampling provides pixels of equal areas, which has many practical advantages.

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

.. image:: https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/spherical_sampling.png
    :width: 700
    :alt: Visualization of spherical sampling schemes
    :align: center

Specifying Sampling Schemes with the ``S2FFT`` API
--------------------------------------------------

Most public ``S2FFT`` functions accept a ``sampling`` optional argument, which will be interpreted as the sampling scheme to be used in that call to the function.
Sampling schemes are specified by providing the corresponding character string to the ``sampling`` argument (listed in the "API string" column of the table above).

Not all functions support all available sampling schemes; the function docstring will clarify which schemes are accepted by the function.

McEwen & Wiaux
--------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{2L-1}, &\quad t\in\lbrace 0,1,...,L-1 \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},     &\quad p\in\lbrace 0,1,...,2L-2\rbrace.

The total number of samples is $N_{MW} = (L-1)(2L-1)+1$.
Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required.
It also requires fewer sampling points than the Gauss-Legendre scheme, though asymptotically the number of sampling points is the same as this scheme.

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^3)`, and the method is stable to band-limits of $L = 4096$.

McEwen & Wiaux SS
-----------------

Driscoll & Healy
----------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi t}{2L},  &\quad t\in\lbrace 0, 1, ..., 2L-1\rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L}, &\quad p\in\lbrace 0, 1, ..., 2L-1\rbrace.

This results in a total of $2L(2L-1) + 1$ sampling points, which are denser near the poles than the equator.

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^2(\log L)^2)`, and the method is stable to band-limits $L$ between 1024 and 2048.

Gauss-Legendre
--------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{2L},  &\quad t\in\lbrace 0, 1, ..., L-1\rbrace, \\
  \varphi_p &= \frac{\pi p}{L},        &\quad p\in\lbrace 0, 1, ...,2L-2\rbrace.

This results in a total of $L(2L-1)$ sampling points.
Note that ``S2FFT`` uses the equiangular Gauss-Legendre sampling scheme, so it is necessary to replace the definition of :math:`\Delta \lambda` in equation (12) of [#gl]_ with

.. math::

  \Delta \lambda = \frac{\pi}{N+1}, \quad j\in\lbrace 0, ..., 2N\rbrace,

for consistency with the notation we use in this section.

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^3)`, and the method is stable to band-limits $L$ between 1024 and 2048.

HEALPix
-------

.. TODO: Could create a citations file since I imagine this is not the only place we'll want to reference 
.. these files: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#citations

.. rubric:: References

.. [#mw] `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_
.. [#mwss] FIXME
.. [#dh] `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_
.. [#gl] `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_
.. [#hp] `Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_