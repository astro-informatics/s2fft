Sampling Schemes
================

.. image:: https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/spherical_sampling.png
    :width: 700
    :alt: Visualization of spherical sampling schemes
    :align: center

The structure of the algorithms implemented in ``S2FFT`` can support a number of sampling schemes.
On this page we give a brief overview of how samples are drawn from the sphere, for each of the sampling schemes that are supported.
A summary of the key differences between the supported sampling schemes is also provided :ref:`in the table below <sampling-comparison-table>`, with further information available in the dedicated section for each scheme.
A more thorough overview of the schemes can be found in section 4.2 of `Price & McEwen (2025) <https://arxiv.org/abs/2311.14670>`_.

All transforms implemented by ``S2FFT`` must be informed of which sampling scheme has been used to draw signal values on the sphere (with the default typically being the :ref:`MW <mcewen-wiaux-mw>` scheme).
This is specified by providing the ``sampling`` argument to the transform in question, when the transform is called.
Other utility functions also accept a ``sampling`` argument, which is used to adjust the behaviour of the function accordingly based on the sampling scheme being used.

As a (somewhat trivial) example to illustrate these conventions, we can generate 'samples' from different schemes from a known signal, and have ``S2FFT`` perform forward transforms on these signals.
First, we perform some setup and imports;

.. code-block:: python

  import jax
  jax.config.update("jax_enable_x64", True)

  import jax.numpy as jnp

  from s2fft.base_transforms.spherical import forward, inverse
  from s2fft.sampling.s2_samples import thetas, phis_equiang

  L = 128

  def signal(theta, phi):
      """Our known signal function, that we will 'sample' from"""
      return jnp.cos(theta + phi)**2

We can generate arrays containing the :math:`\theta_t` and :math:`\varphi_p` coordinates for the MW and GL schemes using :func:`~s2fft.sampling.s2_sampling.thetas` and :func:`~s2fft.sampling.s2_sampling.phis_equiang`.
Passing the ``sampling`` argument to each of these functions specifies which scheme we want to generate sample coordinates for:

.. code-block:: python

  # Generate (theta, phi) points used by the MW scheme
  mw_thetas, mw_phis = thetas(L, sampling="mw"), phis_equiang(L, sampling="mw")
  # Our signal 'sampled' according to the MW scheme
  mw_signal_samples = signal(*jnp.meshgrid(mw_thetas, mw_phis, indexing='ij'))

  # Generate (theta, phi) points used by the GL scheme
  gl_thetas, gl_phis = thetas(L, sampling="gl"), phis_equiang(L, sampling="gl")
  # Our signal 'sampled' according to the MW scheme
  gl_signal_samples = signal(*jnp.meshgrid(gl_thetas, gl_phis, indexing='ij'))

Now that we have samples from two signals, we can forward transform obtain the harmonic coefficients.
In each case, since our sample was "obtained" using a different sampling scheme, we need to specify this to the :func:`~s2fft.base_transforms.spherical.forward` transform when we call it.

.. code-block:: python

  flm_mw = forward(mw_signal_samples, L, sampling="mw")
  flm_gl = forward(gl_signal_samples, L, sampling="gl")
  
  # FIXME: This throws, so my understanding of the package is clearly wrong!
  jnp.assert_allclose(flm_mw, flm_gl)

.. _sampling-comparison-table:

.. list-table:: At-a-glance comparison of sampling schemes
    :header-rows: 1
    :align: center

    * - Scheme
      - API string
      - Equiangular
      - Equal region area
      - Sampling theorem
    * - :ref:`mcewen-wiaux-mw`
      - ``"mw"``
      - Yes
      - No
      - Yes
    * - :ref:`mcewen-wiaux-mwss`
      - ``"mwss"``
      - Yes
      - No
      - Yes
    * - :ref:`driscoll-healy-dh`
      - ``"dh"``
      - Yes
      - No
      - Yes
    * - :ref:`guass-legendre-gl`
      - ``"gl"``
      - Yes
      - No
      - Yes
    * - :ref:`healpix`
      - ``"healpix"``
      - No
      - Yes
      - No

We adopt the usual ``S2FFT`` conventions for spherical coordinates; :math:`\theta\in[0, \pi]` (colatitude) and :math:`\varphi\in[0,2\pi)` (longitude), with :math:`\theta_t` and :math:`\varphi_p` being the discretised samples (indexed by $t$ and $p$) drawn by the sampling scheme.
We denote by $L$ the band-limit of the signals we are considering.

.. _mcewen-wiaux-mw:

McEwen & Wiaux (MW)
-------------------

The MW sampling theorem reduces the Nyquist rate on the sphere by a factor of two compared to the DH approach, halving the number of spherical samples required.

It also requires fewer sampling points than the GL scheme, though asymptotically the number of sampling points used by GL is the same as MW.

Sample positions are defined by

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{2L-1}, &\quad t\in\lbrace 0,1,...,L-1 \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},     &\quad p\in\lbrace 0,1,...,2L-2\rbrace.

The total number of samples is :math:`N_{MW} = (L-1)(2L-1)+1 \sim 2L^2`.

Further information; `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_.

.. _mcewen-wiaux-mwss:

McEwen & Wiaux with Symmetric Sampling (MWSS)
---------------------------------------------

This sampling scheme uses slightly more samples than MW, but still :math:`\sim 2L^2`.
In exchange, the sample locations possess antipodal symmetry.

Sample positions are defined by

.. math::

  \theta_t  &= \frac{2\pi t}{2L},  &\quad t\in\lbrace 0,1,...,L   \rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L},  &\quad p\in\lbrace 0,1,...,2L-1\rbrace.

Further information; `Ocampo, Price, & McEwen (2023) <https://arxiv.org/abs/2209.13603>`_.

.. _driscoll-healy-dh:

Driscoll & Healy (DH)
---------------------

Sample positions are defined by

.. math::

  \theta_t  &= \frac{\pi (2t+1)}{4L},  &\quad t\in\lbrace 0, 1, ..., 2L-1\rbrace, \\
  \varphi_p &= \frac{2\pi p}{2L-1},    &\quad p\in\lbrace 0, 1, ..., 2L-2\rbrace.

This results in :math:`\sim 4L^2` samples on the sphere, which are denser near the poles than the equator.

Further information; `Driscoll & Healy (1995) <https://www.sciencedirect.com/science/article/pii/S0196885884710086>`_, though also see FIXME for the :math:`\theta_t, \varphi_p` scheme adopted here.

.. _guass-legendre-gl:

Gauss-Legendre (GL)
-------------------

The GL sampling theorem also requires :math:`\sim 2L^2` samples on the sphere, though in practice it requires more samples to be placed than MW.

The :math:`\theta_t` are determined by the roots of the Legendre polynomials of order $L$, whilst the :math:`\varphi_p` are defined by

.. math::

  \varphi_p = \frac{2\pi p}{2L-1},  \quad p\in\lbrace 0, 1, ...,2L-2\rbrace.

Further information; `Gauss-Legendre (1986) <https://link.springer.com/article/10.1007/BF02519350>`_.

.. _healpix:

HEALPix
-------

HEALPix sampling provides regions (pixels) of equal areas which can have many practical advantages.

However, HEALPix sampling **does not** exhibit a sampling theorem and so the corresponding harmonic transforms **do not** achieve machine precision but exhibit some error.

A HEALPix grid is defined by a resolution parameter $N_{side}$.
Given a resolution parameter, the grid will contain $N_{hp} = 12 N_{side}^2$ regions of the same area $\frac{\pi}{3N_{side}^2$.
The regions will be laid out on $4N_{side}-1$ iso-latitude rings, and the distribution of regions will be symmetric about the equator.
For the equations defining the exact positioning of the regions, their centres, and their boundaries, see section 5 of `Gorski et al. (2005) <https://arxiv.org/abs/astro-ph/0409513>`_.

Further information; `Gorski et al. (2005) <https://arxiv.org/abs/astro-ph/0409513>`_.
