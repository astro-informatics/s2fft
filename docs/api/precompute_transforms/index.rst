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
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~s2fft.precompute_transforms.spherical.inverse_transform` 
     - Inverse spherical harmonic transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.spherical.inverse_transform_jax`
     - Inverse spherical harmonic transform (JAX)
   * - :func:`~s2fft.precompute_transforms.spherical.forward` 
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~s2fft.precompute_transforms.spherical.forward_transform`
     - Forward spherical harmonic transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.spherical.forward_transform_jax`
     - Forward spherical harmonic transform (JAX)

.. list-table:: Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.wigner.inverse`
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~s2fft.precompute_transforms.wigner.inverse_transform`
     - Inverse Wigner transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.wigner.inverse_transform_jax`
     - Inverse Wigner transform (JAX)
   * - :func:`~s2fft.precompute_transforms.wigner.forward`
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~s2fft.precompute_transforms.wigner.forward_transform`
     - Forward Wigner transform (NumPy)
   * - :func:`~s2fft.precompute_transforms.wigner.forward_transform_jax`
     - Forward Wigner transform (JAX)

.. list-table:: Constructing Kernels for precompute transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.construct.spin_spherical_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform.
   * - :func:`~s2fft.precompute_transforms.construct.wigner_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for Wigner transform.
   * - :func:`~s2fft.precompute_transforms.construct.healpix_phase_shifts`
     - Builds a vector of corresponding phase shifts for each HEALPix latitudinal ring.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Precompute Transforms

   construct
   spin_spherical 
   wigner

