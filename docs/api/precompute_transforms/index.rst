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

.. list-table:: Fourier-Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.fourier_wigner.inverse_transform`
     - Inverse Wigner transform with Fourier method (NumPy)
   * - :func:`~s2fft.precompute_transforms.fourier_wigner.inverse_transform_jax`
     - Inverse Wigner transform with Fourier method (JAX)
   * - :func:`~s2fft.precompute_transforms.fourier_wigner.forward_transform`
     - Forward Wigner transform with Fourier method (NumPy)
   * - :func:`~s2fft.precompute_transforms.fourier_wigner.forward_transform_jax`
     - Forward Wigner transform with Fourier method (JAX)
    
.. list-table:: Custom Operations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.custom_ops.wigner_subset_to_s2`
     - Transforms an arbitrary subset of Wigner coefficients onto a subset of spin signals on the sphere.
   * - :func:`~s2fft.precompute_transforms.custom_ops.wigner_subset_to_s2_jax`
     - Transforms an arbitrary subset of Wigner coefficients onto a subset of spin signals on the sphere (JAX).
   * - :func:`~s2fft.precompute_transforms.custom_ops.so3_to_wigner_subset`
     - Transforms a signal on the rotation group to an arbitrary subset of its Wigner coefficients.
   * - :func:`~s2fft.precompute_transforms.custom_ops.so3_to_wigner_subset_jax`
     - Transforms a signal on the rotation group to an arbitrary subset of its Wigner coefficients (JAX).
   * - :func:`~s2fft.precompute_transforms.custom_ops.s2_to_wigner_subset`
     - Transforms from a collection of arbitrary spin signals on the sphere to the corresponding collection of their harmonic coefficients.
   * - :func:`~s2fft.precompute_transforms.custom_ops.s2_to_wigner_subset_jax`
     - Transforms from a collection of arbitrary spin signals on the sphere to the corresponding collection of their harmonic coefficients (JAX).

.. list-table:: Constructing Kernels for precompute transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.precompute_transforms.construct.spin_spherical_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform.
   * - :func:`~s2fft.precompute_transforms.construct.wigner_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for Wigner transform.
   * - :func:`~s2fft.precompute_transforms.construct.spin_spherical_kernel_jax`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform (JAX).
   * - :func:`~s2fft.precompute_transforms.construct.wigner_kernel_jax`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for Wigner transform (JAX).
   * - :func:`~s2fft.precompute_transforms.construct.fourier_wigner_kernel`
     - Builds a kernel including quadrature weights and Fourier coefficienfs of Wigner d-functions
   * - :func:`~s2fft.precompute_transforms.construct.fourier_wigner_kernel_jax`
     - Builds a kernel including quadrature weights and Fourier coefficienfs of Wigner d-functions (JAX).
   * - :func:`~s2fft.precompute_transforms.construct.healpix_phase_shifts`
     - Builds a vector of corresponding phase shifts for each HEALPix latitudinal ring.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Precompute Transforms

   construct
   spin_spherical 
   wigner
   fourier_wigner
   custom_ops

