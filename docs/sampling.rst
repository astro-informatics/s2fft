Sampling Schemes
================

.. FIXME give the table a reference!

The structure of the algorithms implemented in ``S2FFT`` can support a number of sampling schemes.
On this page we give a brief overview and comparison of the sampling schemes that are supported, which largely follows the review in section 4.2 of [#price_mcewen_2025]_ .
A summary of the key differences between the supported sampling schemes is also provided in the table below, with further information available in the dedicated section for each scheme.

Most public ``S2FFT`` functions accept a ``sampling`` optional argument, which will be interpreted as the sampling scheme to be used in that call to the function.
Sampling schemes are specified by providing the corresponding character string to the ``sampling`` argument (see the table below).
Not all functions support all available sampling schemes; the function docstring will clarify which schemes are accepted by the function.

.. list-table:: At-a-glance comparison of sampling schemes
    :header-rows: 1
    :align: center

    * - Scheme
      - API string
      - Equiangular
      - Equal region area
      - Has sampling theorem
    * - McEwen & Wiaux
      - ``"mw"``
      - Yes
      - No
      - Yes
    * - McEwen & Wiaux SS
      - ``"mwss"``
      - Yes
      - No
      - Yes
    * - Driscoll & Healy
      - ``"dh"``
      - Yes
      - No
      - Yes
    * - Gauss-Legendre
      - ``"gl"``
      - No
      - No
      - Yes
    * - HEALPix
      - ``"healpix"``
      - No
      - Yes
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

McEwen & Wiaux (MW)
-------------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{2L-1}, &\quad t\in\lbrace 0,1,...,L-1 \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},     &\quad p\in\lbrace 0,1,...,2L-2\rbrace.

The total number of samples is :math:`N_{MW} = (L-1)(2L-1)+1 \sim 2L^2`.
Note that the McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the Driscoll & Healy approach, halving the number of spherical samples required.
It also requires fewer sampling points than the Gauss-Legendre scheme, though asymptotically the number of sampling points is the same as this scheme.

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^3)`, and the method is stable to band-limits of $L = 4096$.

McEwen & Wiaux with Symmetric Sampling (MWSS)
---------------------------------------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{2\pi t}{2L},  &\quad t\in\lbrace 0,1,...,L   \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L},  &\quad p\in\lbrace 0,1,...,2L-1\rbrace.

This sampling scheme uses slightly more, but still :math:`\sim 2L^2`, samples than MW.
In exchange, the sample locations possess antipodal symmetry.

Driscoll & Healy (DH)
---------------------

Samples are placed at positions :math:`(\theta_t, \varphi_p)` where

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{4L},  &\quad t\in\lbrace 0, 1, ..., 2L-1\rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},    &\quad p\in\lbrace 0, 1, ..., 2L-2\rbrace.

This results in :math:`\sim 4L^2` samples on the sphere, which are denser near the poles than the equator.

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^2(\log L)^2)`, however not all variants of such algorithms are universally stable.
The so-called "semi-naive" algorithm is universally stable, but has complexity :math:`\mathcal{O}(L^3)`.

The method is stable to band-limits $L$ between 1024 and 2048.

Gauss-Legendre (GL)
-------------------

Samples positions in $\theta$ are determined by the roots of the Legendre polynomials of order $L$, whilst positions in :math:`\varphi_p` are defined by

.. math::

  \varphi_p &= \frac{2\pi p}{2L-1},        &\quad p\in\lbrace 0, 1, ...,2L-2\rbrace.

The GL sampling theorem also requires :math:`\sim 2L^2` samples on the sphere (though in practice it requires slightly more samples than MW).

Complexity for forward/inverse transforms is :math:`\mathcal{O}(L^3)`, and the method is stable to band-limits $L$ between 1024 and 2048.

HEALPix
-------

A HEALPix grid is defined by a resolution parameter $N_{side}$.
Given a resolution parameter, the grid will contain $N_{hp} = 12 N_{side}^2$ regions of the same area $\frac{\pi}{3N_{side}^2$.
The regions will be laid out on $4N_{side}-1$ iso-latitude rings, and the distribution of regions will be symmetric about the equator.
For the equations defining the exact positioning of the regions, their centres, and their boundaries, see section 5 of [#hp]_ .

HEALPix sampling is not equiangular, but does provide pixels of equal areas which can have many practical advantages.
However, HEALPix sampling **does not** exhibit a sampling theorem and so the corresponding harmonic transforms **do not** achieve machine precision but exhibit some error.

.. TODO: Could create a citations file since I imagine this is not the only place we'll want to reference 
.. these files: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#citations

.. rubric:: References

.. [#mw] `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_
.. [#price_mcewen_2025] `Price & McEwen (2025) <https://arxiv.org/abs/2311.14670>`_
.. [#mwss] `Ocampo, Price, & McEwen (2023) <https://arxiv.org/abs/2209.13603>`_
.. [#dh] `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_
.. [#gl] `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_
.. [#hp] `Gorski et al. 2005 <https://arxiv.org/abs/astro-ph/0409513>`_