:html_theme.sidebar_secondary.remove:

**************************
Wigner-d recursions
**************************

.. list-table:: Price-McEwen recursion functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.recursions.price_mcewen.compute_all_slices`
     - Computes all necessary slices of Wigner-d planes using Price-McEwen recursion (NumPy).
   * - :func:`~s2fft.recursions.price_mcewen.compute_all_slices_jax`
     - Computes all necessary slices of Wigner-d planes using Price-McEwen recursion (JAX).
   * - :func:`~s2fft.recursions.price_mcewen.generate_precomputes`
     - Deprecated: constructs the :math:`\mathcal{O}(L^2)` precomputes for the spin-spherical harmonic transform, which the Price-McEwen recursion no longer uses (NumPy).
   * - :func:`~s2fft.recursions.price_mcewen.generate_precomputes_jax`
     - Deprecated: constructs the :math:`\mathcal{O}(L^2)` precomputes for the spin-spherical harmonic transform, which the Price-McEwen recursion no longer uses (JAX).
   * - :func:`~s2fft.recursions.price_mcewen.generate_precomputes_wigner`
     - Deprecated: constructs the :math:`\mathcal{O}(NL^2)` precomputes for the Wigner transform, which the Price-McEwen recursion no longer uses (NumPy).
   * - :func:`~s2fft.recursions.price_mcewen.generate_precomputes_wigner_jax`
     - Deprecated: constructs the :math:`\mathcal{O}(NL^2)` precomputes for the Wigner transform, which the Price-McEwen recursion no longer uses (JAX).

.. list-table:: Turok-Bucher recursion functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.recursions.turok.compute_full`
     - Compute the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher recursion.
   * - :func:`~s2fft.recursions.turok.compute_slice`
     - Compute a particular slice :math:`m^{\prime}`, denoted `mm`, of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher recursion.
   * - :func:`~s2fft.recursions.turok.compute_quarter_slice`
     - Compute a single slice at :math:`m^{\prime}` of the Wigner-d matrix evaluated at :math:`\beta`.
   * - :func:`~s2fft.recursions.turok.compute_quarter`
     - Compute the left quarter triangle of the Wigner-d matrix via Turok & Bucher recursion.
   * - :func:`~s2fft.recursions.turok.fill`
     - Reflects Wigner-d quarter plane to complete full matrix by using symmetry properties of the Wigner-d matrices.
   * - :func:`~s2fft.recursions.turok_jax.compute_slice` (JAX)
     - Compute a particular slice :math:`m^{\prime}`, denoted `mm`, of the complete Wigner-d matrix at polar angle :math:`\beta` using Turok & Bucher recursion (JAX).
   * - :func:`~s2fft.recursions.turok_jax.reindex` (JAX)
     - Reorders indexing of Wigner-d matrix, only necessary to maintain fixed length JAX arrays.

.. list-table:: Trapani recursion functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.recursions.trapani.compute_eighth`
     - Compute Wigner-d at argument :math:`\pi/2` for eighth of plane using Trapani & Navaza recursion.
   * - :func:`~s2fft.recursions.trapani.compute_quarter_vectorized`
     - Compute Wigner-d at argument :math:`\pi/2` for quarter of plane using Trapani & Navaza recursion (vector implementation).
   * - :func:`~s2fft.recursions.trapani.compute_quarter_jax`
     - Compute Wigner-d at argument :math:`\pi/2` for quarter of plane using Trapani & Navaza recursion (JAX implementation).
   * - :func:`~s2fft.recursions.trapani.fill_eighth2quarter`
     - Fill in quarter of Wigner-d plane from eighth.
   * - :func:`~s2fft.recursions.trapani.fill_quarter2half`
     - Fill in half of Wigner-d plane from quarter.
   * - :func:`~s2fft.recursions.trapani.fill_quarter2half_vectorized`
     - Fill in half of Wigner-d plane from quarter (vectorised implementation).
   * - :func:`~s2fft.recursions.trapani.fill_quarter2half_jax`
     - Fill in half of Wigner-d plane from quarter (JAX implementation).
   * - :func:`~s2fft.recursions.trapani.fill_half2full`
     - Fill in full Wigner-d plane from half.
   * - :func:`~s2fft.recursions.trapani.fill_half2full_vectorized`
     - Fill in full Wigner-d plane from half (vectorized implementation).
   * - :func:`~s2fft.recursions.trapani.fill_half2full_jax`
     - Fill in full Wigner-d plane from half (JAX implementation).
   * - :func:`~s2fft.recursions.trapani.compute_full`
     - Compute Wigner-d at argument :math:`\pi/2` for full plane using Trapani & Navaza recursion (multiple implementations).
   * - :func:`~s2fft.recursions.trapani.compute_full_loop`
     - Compute Wigner-d at argument :math:`\pi/2` for full plane using Trapani & Navaza recursion (loop-based implementation).
   * - :func:`~s2fft.recursions.trapani.compute_quarter`
     - Compute Wigner-d at argument :math:`\pi/2` for quarter plane using Trapani & Navaza recursion.
   * - :func:`~s2fft.recursions.trapani.compute_full_vectorized`
     - Compute Wigner-d at argument :math:`\pi/2` for full plane using Trapani & Navaza recursion (vectorized implementation).
   * - :func:`~s2fft.recursions.trapani.compute_full_jax`
     - Compute Wigner-d at argument :math:`\pi/2` for full plane using Trapani & Navaza recursion (JAX implementation).

.. list-table:: Risbo recursion functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.recursions.risbo.compute_full`
     - Compute Wigner-d at argument :math:`\beta` for full plane using Risbo recursion.
   * - :func:`~s2fft.recursions.risbo_jax.compute_full`
     - Compute Wigner-d at argument :math:`\beta` for full plane using Risbo recursion (JAX implementation).

.. warning:: 

      The primary recursion used by ``S2FFT`` is the Price-McEwen recursion, though 
      we include other popular recursions for comparison. One should however note that 
      the development time for these recursions was minimal, thus functionality for these 
      recursions is very limited.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Precompute Transforms

   price_mcewen
   risbo
   risbo_jax
   trapani
   turok
   turok_jax