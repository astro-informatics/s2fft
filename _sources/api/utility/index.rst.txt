:html_theme.sidebar_secondary.remove:

**************************
Utility Functions
**************************

.. list-table:: HEALPix fast Fourier transform planning functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.utils.healpix_ffts.healpix_ifft`
     - Wrapper function for the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing.
   * - :func:`~s2fft.utils.healpix_ffts.healpix_ifft_numpy`
     - Computes the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing (NumPy).
   * - :func:`~s2fft.utils.healpix_ffts.healpix_ifft_jax`
     - Computes the Inverse Fast Fourier Transform with spectral folding in the polar regions to mitigate aliasing (JAX).
   * - :func:`~s2fft.utils.healpix_ffts.healpix_fft`
     - Wrapper function for the Forward Fast Fourier Transform with spectral back-projection in the polar regions to manually enforce Fourier periodicity.
   * - :func:`~s2fft.utils.healpix_ffts.healpix_fft_numpy`
     - Computes the Forward Fast Fourier Transform with spectral back-projection in the polar regions (NumPy).
   * - :func:`~s2fft.utils.healpix_ffts.healpix_fft_jax`
     - Computes the Forward Fast Fourier Transform with spectral back-projection in the polar regions (NumPy).
   * - :func:`~s2fft.utils.healpix_ffts.spectral_folding`
     - Folds higher frequency Fourier coefficients back onto lower frequency coefficients (NumPy).
   * - :func:`~s2fft.utils.healpix_ffts.spectral_folding_jax`
     - Folds higher frequency Fourier coefficients back onto lower frequency coefficients (JAX).
   * - :func:`~s2fft.utils.healpix_ffts.spectral_periodic_extension`
     - Extends lower frequency Fourier coefficients onto higher frequency coefficients (NumPy).
   * - :func:`~s2fft.utils.healpix_ffts.spectral_periodic_extension_jax`
     - Extends lower frequency Fourier coefficients onto higher frequency coefficients (JAX).
   

.. list-table:: Quadrature functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.utils.quadrature.quad_weights_transform`
     - Compute quadrature weights for :math:`\theta` and :math:`\phi` integration *to use in transform* for various sampling schemes.
   * - :func:`~s2fft.utils.quadrature.quad_weights`
     - Compute quadrature weights for :math:`\theta` and :math:`\phi` integration for various sampling schemes.
   * - :func:`~s2fft.utils.quadrature.quad_weights_hp`
     - Compute HEALPix quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~s2fft.utils.quadrature.quad_weights_dh`
     - Compute DH quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~s2fft.utils.quadrature.quad_weights_mw`
     - Compute MW quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~s2fft.utils.quadrature.quad_weights_mwss`
     - Compute MWSS quadrature weights for :math:`\theta` and :math:`\phi` integration.
   * - :func:`~s2fft.utils.quadrature.quad_weight_dh_theta_only`
     - Compute DH quadrature weight for :math:`\theta` integration (only), for given :math:`\theta`.
   * - :func:`~s2fft.utils.quadrature.quad_weights_mw_theta_only`
     - Compute MW quadrature weights for :math:`\theta` integration (only).
   * - :func:`~s2fft.utils.quadrature.quad_weights_mwss_theta_only`
     - Compute MWSS quadrature weights for :math:`\theta` integration (only).
   * - :func:`~s2fft.utils.quadrature.mw_weights`
     - Compute MW weights given as a function of index m.

.. note::

      JAX versions of these functions share an almost identical function trace and 
      are simply accessed by the sub-module :func:`~s2fft.utils.quadrature_jax`.

.. list-table:: Periodic resampling functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.utils.resampling.periodic_extension`
     - Perform period extension of MW/MWSS signal on the sphere in harmonic domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
   * - :func:`~s2fft.utils.resampling.periodic_extension_spatial_mwss`
     - Perform period extension of MWSS signal on the sphere in spatial domain, extending :math:`\theta` domain from :math:`[0,\pi]` to :math:`[0,2\pi]`.
   * - :func:`~s2fft.utils.resampling.upsample_by_two_mwss`
     - Upsample MWSS sampled signal on the sphere defined on domain :math:`[0,\pi]` by a factor of two.
   * - :func:`~s2fft.utils.resampling.upsample_by_two_mwss_ext`
     - Upsample an extended MWSS sampled signal on the sphere defined on domain :math:`[0,2\pi]` by a factor of two.
   * - :func:`~s2fft.utils.resampling.downsample_by_two_mwss`
     - Downsample an MWSS sampled signal on the sphere.
   * - :func:`~s2fft.utils.resampling.unextend`
     - Unextend MW/MWSS sampled signal from :math:`\theta` domain :math:`[0,2\pi]` to :math:`[0,\pi]`.
   * - :func:`~s2fft.utils.resampling.mw_to_mwss_phi`
     - Convert :math:`\phi` component of signal on the sphere from MW sampling to MWSS sampling.
   * - :func:`~s2fft.utils.resampling.mw_to_mwss_theta`
     - Convert :math:`\theta` component of signal on the sphere from MW sampling to MWSS sampling.
   * - :func:`~s2fft.utils.resampling.mw_to_mwss`
     - Convert signal on the sphere from MW sampling to MWSS sampling.

.. list-table:: Signal generating functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.utils.signal_generator.generate_flm`
     - Generate a 2D set of random harmonic coefficients.
   * - :func:`~s2fft.utils.signal_generator.generate_flmn`
     - Generate a 3D set of random Wigner coefficients.


.. note::

      JAX versions of these functions share an almost identical function trace and 
      are simply accessed by the sub-module :func:`~s2fft.utils.resampling_jax`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Utility Functions

   signal_generator
   resampling
   resampling_jax
   quadrature
   quadrature_jax
   healpix_ffts
   utils
   logs
   