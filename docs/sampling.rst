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

Specifying Sampling Schemes with the ``S2FFT`` API
--------------------------------------------------

Most public ``S2FFT`` functions accept a ``sampling`` optional argument, which will be interpreted as the sampling scheme to be used in that call to the function.
Sampling schemes are specified by providing the corresponding character string to the ``sampling`` argument:

* ``"mw"``; `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_.
* ``"mwss"``;
* ``"dh"``; `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_
* ``"gl"``; `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_
* ``"healpix"``; `HEALPix <https://healpix.jpl.nasa.gov>`_

Not all functions support all available sampling schemes; the function docstring will clarify which schemes are accepted by the function.

FIXME add an example for simple inverse / forward? Use some code from the tests to illustrate...?

Comparison Of Sampling Schemes
------------------------------

MW:

- $\theta_t = \frac{\pi (2t+1)}{2L-1}$, $t\in\lbrace 0,1,...,L-1\rbrace$.
- $\varphi_p = \frac{2\pi p}{2L-1}$, $p\in\lbrace 0,1,...,2L-2\rbrace$.
- $N_{MW} = (L-1)(2L-1)+1$, so order $2L^2$ samples on the sphere.
- Requires symmetric sampling in $\theta$ about the South pole; repeat samples at the poles are eliminated, but the $\theta=\pi$ repeated sample cannot be eliminated since we need a discretisation that is symmetric about $\pi$ but which also contains an odd number of points.

Requires less than half the number of samples to represents a band-limited signal on the sphere exactly, compared to other equiangular sampling theorems. Requires asymptotically the same, but smaller, number of samples than GL.

Complexity for forward/inverse transforms is $O(L^3)$.

Stable to $L = 4096$.

GL:

$N = 2L-1$ I think....(paper notation translation)
- $\theta_t = \frac{\pi (t + \frac{1}{2})}{2L}$
- $\varphi_p = \frac{2\pi p}{2L}$
- Sampling theorem requires order $2L^2$ samples

Complexity for forward/inverse transforms is $O(L^3)$?

Go unstable between $L = 1024$ and $L = 2048$.

DH:

Sample points are denser near the poles (than the equator), and so the sample points must be weighted to reflect this.

- $\theta_t = \frac{\pi t}{2L}$, $t\in\lbrace 0, 1, ..., 2L-1\rbrace$
- $\varphi_p = \frac{2\pi p}{2L}$, $t\in\lbrace 0, 1, ..., 2L-1\rbrace$
- Sampling theorem requires order $4L^2$ samples
- Paper claims that we can transform in $O(L^2(\log L)^2)$, this would make it asymptotically faster than MW? MW paper does quote approx 25% slower than DH, so maybe this is to be expected?

Go unstable between $L = 1024$ and $L = 2048$.