:html_theme.sidebar_secondary.remove:

**************************
Reference Transforms
**************************

.. list-table:: Spherical harmonic transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.base_transforms.spherical.inverse`
     - Wrapper function around various versions of classic inverse spherical harmonic transforms.
   * - :func:`~s2fft.base_transforms.spherical.forward` 
     - Wrapper function around various versions of classic forward spherical harmonic transforms.


.. list-table:: Wigner transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2fft.base_transforms.wigner.inverse`
     - Classic algorithm for forward Wigner transform.
   * - :func:`~s2fft.base_transforms.wigner.forward`
     - Classic algorithm for inverse Wigner transform.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Base Transforms

   spin_spherical_transform
   wigner_transform
