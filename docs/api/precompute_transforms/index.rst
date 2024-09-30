:html_theme.sidebar_secondary.remove:

**************************
Precompute Functions
**************************

.. list-table:: Spherical harmonic transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.spherical.inverse`
     - Wrapper function around NumPy/JAX/Torch inverse methods
   * - :func:`~s2fft.precompute_transforms.spherical.inverse_transform` 
     - Inverse spherical harmonic transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.spherical.inverse_transform_jax`
     - Inverse spherical harmonic transform (JAX)
   * - :func:`~s2fft.precompute_transforms.spherical.inverse_transform_torch`
     - Inverse spherical harmonic transform (Torch)
   * - :func:`~s2fft.precompute_transforms.spherical.forward` 
     - Wrapper function around NumPy/JAX/Torch forward methods
   * - :func:`~s2fft.precompute_transforms.spherical.forward_transform`
     - Forward spherical harmonic transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.spherical.forward_transform_jax`
     - Forward spherical harmonic transform (JAX)
   * - :func:`~s2fft.precompute_transforms.spherical.forward_transform_torch`
     - Forward spherical harmonic transform (Torch)

.. list-table:: Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.wigner.inverse`
     - Wrapper function around NumPy/JAX/Torch inverse methods
   * - :func:`~s2fft.precompute_transforms.wigner.inverse_transform`
     - Inverse Wigner transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.wigner.inverse_transform_jax`
     - Inverse Wigner transform (JAX)
   * - :func:`~s2fft.precompute_transforms.wigner.inverse_transform_torch`
     - Inverse Wigner transform (Torch)
   * - :func:`~s2fft.precompute_transforms.wigner.forward`
     - Wrapper function around NumPy/JAX/Torch forward methods
   * - :func:`~s2fft.precompute_transforms.wigner.forward_transform`
     - Forward Wigner transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.wigner.forward_transform_jax`
     - Forward Wigner transform (JAX)
   * - :func:`~s2fft.precompute_transforms.wigner.forward_transform_torch`
     - Forward Wigner transform (Torch)

.. list-table:: Constructing Kernels for precompute transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.construct.spin_spherical_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform.
   * - :func:`~s2fft.precompute_transforms.construct.wigner_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for Wigner transform.
   * - :func:`~s2fft.precompute_transforms.alt_construct.spin_spherical_kernel`
     - Builds a high-spin stable kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform.
   * - :func:`~s2fft.precompute_transforms.alt_construct.wigner_kernel`
     - Builds a high-spin stable kernel including quadrature weights and Wigner-D coefficients for Wigner transform.
   * - :func:`~s2fft.precompute_transforms.construct.healpix_phase_shifts`
     - Builds a vector of corresponding phase shifts for each HEALPix latitudinal ring.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Precompute Transforms

   construct
   alt_construct
   spin_spherical 
   wigner

