:html_theme.sidebar_secondary.remove:

**************************
Transforms
**************************
.. list-table:: Spherical harmonic transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.transforms.spherical.inverse`
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~s2fft.transforms.spherical.inverse_numpy`
     - Inverse spherical harmonic transform (NumPy)
   * - :func:`~s2fft.transforms.spherical.inverse_jax`
     - Inverse spherical harmonic transform (JAX)
   * - :func:`~s2fft.transforms.spherical.forward`
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~s2fft.transforms.spherical.forward_numpy`
     - Forward spherical harmonic transform (NumPy)
   * - :func:`~s2fft.transforms.spherical.forward_jax`
     - Forward spherical harmonic transform (JAX)

.. list-table:: Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.transforms.wigner.inverse`
     - Wrapper function around NumPy/JAX inverse methods
   * - :func:`~s2fft.transforms.wigner.inverse_numpy`
     - Inverse Wigner transform (NumPy)
   * - :func:`~s2fft.transforms.wigner.inverse_jax`
     - Inverse Wigner transform (JAX)
   * - :func:`~s2fft.transforms.wigner.forward`
     - Wrapper function around NumPy/JAX forward methods
   * - :func:`~s2fft.transforms.wigner.forward_numpy`
     - Forward Wigner transform (NumPy)
   * - :func:`~s2fft.transforms.wigner.forward_jax`
     - Forward Wigner transform (JAX)

.. list-table:: On-the-fly Price-McEwen recursions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.transforms.otf_recursions.inverse_latitudinal_step`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (NumPy)
   * - :func:`~s2fft.transforms.otf_recursions.inverse_latitudinal_step_jax`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (JAX)
   * - :func:`~s2fft.transforms.otf_recursions.forward_latitudinal_step`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (NumPy)
   * - :func:`~s2fft.transforms.otf_recursions.forward_latitudinal_step_jax`
     - On-the-fly wigner-recursion for latitudinal portion of overall transform (JAX)

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Transforms

   on_the_fly_recursions
   spin_spherical_transform
   wigner

