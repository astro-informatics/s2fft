:html_theme.sidebar_secondary.remove:

**************************
Precompute Functions
**************************

.. list-table:: Spherical harmonic transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~inverse`
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~inverse_numpy` 
     - Inverse spherical harmonic transform (NumPy)
   * - :func:`~inverse_jax`
     - Inverse spherical harmonic transform (JAX)
   * - :func:`~forward` 
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~forward_numpy`
     - Forward spherical harmonic transform (NumPy)
   * - :func:`~forward_jax`
     - Forward spherical harmonic transform (JAX)

.. list-table:: Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~inverse`
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~inverse_numpy`
     - Inverse Wigner transform (NumPy)
   * - :func:`~inverse_jax`
     - Inverse Wigner transform (JAX)
   * - :func:`~forward`
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~forward_numpy`
     - Forward Wigner transform (NumPy)
   * - :func:`~forward_jax`
     - Forward Wigner transform (JAX)

.. list-table:: Constructing Kernels for precompute transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~spin_spherical_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for spherical harmonic transform.
   * - :func:`~wigner_kernel`
     - Builds a kernel including quadrature weights and Wigner-D coefficients for Wigner transform.
   * - :func:`~healpix_phase_shifts`
     - Builds a vector of corresponding phase shifts for each HEALPix latitudinal ring.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Precompute Transforms

   construct
   spin_spherical 
   wigner

