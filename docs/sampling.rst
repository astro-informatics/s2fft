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

McEwen & Wiaux
--------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{2L-1}, &\quad t\in\lbrace 0,1,...,L-1 \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},     &\quad p\in\lbrace 0,1,...,2L-2\rbrace.

The total number of samples is $N_{MW} = (L-1)(2L-1)+1$.
This sampling scheme requires symmetric sampling in $\theta$ about the South pole; repeat samples at the poles are eliminated, but the $\theta=\pi$ repeated sample cannot be eliminated since a discretisation `with an odd number of points` that is symmetric about $\pi$ is needed.

This scheme requires less than half the number of samples to represents a band-limited signal on the sphere exactly, compared to other equiangular sampling theorems.
The exception being for the Gauss-Legendre scheme, which requires asymptotically the same number of sampling points (though GL still requires more samples).

Complexity for forward/inverse transforms is $O(L^3)$, and the method is stable to band-limits of $L = 4096$.

FIXME MWSS as a subsection here?

Driscoll & Healy
----------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi t}{2L},  &\quad t\in\lbrace 0, 1, ..., 2L-1\rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L}, &\quad p\in\lbrace 0, 1, ..., 2L-1\rbrace.

This results in a total of $(2L-1)^2$ sampling points, which are denser near the poles than the equator.

Complexity for forward/inverse transforms is $O(L^2(\log L)^2)$, and the method is stable to band-limits $L$ between 1024 and 2048.

Gauss-Legendre
--------------
.. GL:

.. $N = 2L-1$ I think....(paper notation translation)
.. - $\theta_t = \frac{\pi (t + \frac{1}{2})}{2L}$
.. - $\varphi_p = \frac{2\pi p}{2L}$
.. - Sampling theorem requires order $2L^2$ samples

.. Complexity for forward/inverse transforms is $O(L^3)$?

.. Go unstable between $L = 1024$ and $L = 2048$.
