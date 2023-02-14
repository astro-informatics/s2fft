:html_theme.sidebar_secondary.remove:

**************************
Utility Functions
**************************

.. list-table:: HEALPix fast Fourier transform planning functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~healpix_ifft`
     - Wrapper function for the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing.
   * - :func:`~healpix_ifft_numpy`
     - Computes the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing (NumPy).
   * - :func:`~healpix_ifft_jax`
     - Computes the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing (JAX).
   * - :func:`~healpix_fft`
     - Wrapper function for the Forward Fast Fourier Transform with spectral back-projection in the polar regions to manually enforce Fourier periodicity.
   * - :func:`~healpix_fft_numpy`
     - Computes the Forward Fast Fourier Transform with spectral back-projection in the polar regions (NumPy).
   * - :func:`~healpix_fft_jax`
     - Computes the Forward Fast Fourier Transform with spectral back-projection in the polar regions (NumPy).
   * - :func:`~spectral_folding`
     - Folds higher frequency Fourier coefficients back onto lower frequency coefficients (NumPy).
   * - :func:`~spectral_folding_jax`
     - Folds higher frequency Fourier coefficients back onto lower frequency coefficients (JAX).
   * - :func:`~spectral_periodic_extension`
     - Extends lower frequency Fourier coefficients onto higher frequency coefficients (NumPy).
   * - :func:`~spectral_periodic_extension_jax`
     - Extends lower frequency Fourier coefficients onto higher frequency coefficients (JAX).
   

.. list-table:: Quadrature functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~quad_weights_transform`
     - Compute quadrature weights for :math:`\theta` and :math:`\phi` integration *to use in transform* for various sampling schemes.
   * - :func:`~quad_weights`
     - Compute quadrature weights for :math:`\theta` and :math:`\phi` integration for various sampling schemes.
   * - :func:`~quad_weights_hp`
     - Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~quad_weights_dh`
     - Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~quad_weights_mw`
     - Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~quad_weights_mwss`
     - Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~quad_weight_dh_theta_only`
     - Compute DH quadrature weight for :math:`\theta` integration (only), for given :math:`\theta`.
   * - :func:`~quad_weights_mw_theta_only`
     - Compute MW quadrature weights for :math:`\theta` integration (only).
   * - :func:`~quad_weights_mwss_theta_only`
     - Compute MWSS quadrature weights for :math:`\theta` integration (only).
   * - :func:`~mw_weights`
     - Compute MW weights given as a function of index m.

.. note::

      JAX versions of these functions share an almost identical function trace and 
      are simply accessed by the sub-module :func:`~quadrature_jax`.

.. list-table:: Periodic resampling functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~periodic_extension`
     - Perform period extension of MW/MWSS signal on the sphere in harmonic domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
   * - :func:`~periodic_extension_spatial_mwss`
     - Perform period extension of MWSS signal on the sphere in spatial domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
   * - :func:`~upsample_by_two_mwss`
     - Upsample MWSS sampled signal on the sphere defined on domain :math:`[0,\pi]` by a factor of two.
   * - :func:`~upsample_by_two_mwss_ext`
     - Upsample an extended MWSS sampled signal on the sphere defined on domain :math:`[0,2\pi]` by a factor of two.
   * - :func:`~downsample_by_two_mwss`
     - Downsample an MWSS sampled signal on the sphere.
   * - :func:`~unextend`
     - Unextend MW/MWSS sampled signal from :math:`\theta` domain :math:`[0,2\pi]` to :math:`[0,\pi]`.
   * - :func:`~mw_to_mwss_phi`
     - Convert :math:`\phi` component of signal on the sphere from MW sampling to MWSS sampling.
   * - :func:`~mw_to_mwss_theta`
     - Convert :math:`\theta` component of signal on the sphere from MW sampling to MWSS sampling.
   * - :func:`~mw_to_mwss`
     - Convert signal on the sphere from MW sampling to MWSS sampling.

.. list-table:: Signal generating functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~generate_flm`
     - Generate a 2D set of random harmonic coefficients.
   * - :func:`~generate_flmn`
     - Generate a 3D set of random Wigner coefficients.


.. note::

      JAX versions of these functions share an almost identical function trace and 
      are simply accessed by the sub-module :func:`~resampling_jax`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Utility Functions

   resampling
   resampling_jax
   quadrature
   quadrature_jax
   healpix_ffts
   utils
   logs
   