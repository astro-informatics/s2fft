:html_theme.sidebar_secondary.remove:

**************************
Transforms
**************************
.. list-table:: Spherical harmonic transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.transform.spin_spherical.inverse`
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

.. list-table:: On-the-fly Price-McEwen recursions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~inverse_latitudinal_step`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (NumPy)
   * - :func:`~inverse_latitudinal_step_jax`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (JAX)
   * - :func:`~forward_latitudinal_step`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (NumPy)
   * - :func:`~forward_latitudinal_step_jax`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (JAX)

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Transforms

   on_the_fly_recursions
   spin_spherical_transform
   wigner

