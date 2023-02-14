:html_theme.sidebar_secondary.remove:

**************************
Sampling Functions
**************************
.. list-table:: Array shaping functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~f_shape`
     - Shape of a signal on the sphere and rotation group.
   * - :func:`~flm_shape`
     - Standard shape of harmonic coefficients.
   * - :func:`~flmn_shape`
     - Standard shape of Wigner coefficients.
   * - :func:`~ftm_shape`
     - Shape of intermediate array, before/after latitudinal step.
   * - :func:`~fnab_shape`
     - Shape of intermediate array, before/after latitudinal step.

.. list-table:: Spherical sampling functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~ncoeff`
     - Number of spherical harmonic coefficients for given band-limit L.
   * - :func:`~ntheta`
     - Total number of latitudinal samples.
   * - :func:`~ntheta_extension`
     - Total number of latitudinal samples for periodically extended sampling.
   * - :func:`~thetas`
     - Compute :math:`\theta` samples for given sampling scheme.
   * - :func:`~t2theta`
     - Convert index to :math:`\theta` angle for sampling scheme.
   * - :func:`~nphi_equiang`
     - Total number of longitudinal samples for equiangular sampling schemes.
   * - :func:`~nphi_equitorial_band`
     - Number of :math:`\phi` samples within the equitorial band for HEALPix sampling scheme.
   * - :func:`~nphi_ring`
     - Number of :math:`\phi` samples for HEALPix sampling on given :math:`\theta` ring.
   * - :func:`~phis_ring`
     - Compute :math:`\phi` samples for given :math:`\theta` HEALPix ring.
   * - :func:`~p2phi_ring`
     - Convert index to :math:`\phi` angle for HEALPix for given :math:`\theta` ring.
   * - :func:`~phis_equiang`
     - Compute :math:`\phi` samples for equiangular sampling scheme.
   * - :func:`~ring_phase_shift_hp`
     - Generates a phase shift vector for HEALPix for a given :math:`\theta` ring.

.. list-table:: Pixel indexing functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~elm2ind`
     - Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.
   * - :func:`~ind2elm`
     - Convert from 1D spherical harmonic index to 2D index of :math:`(\ell,m)`.
   * - :func:`~elmn2ind`
     - Convert from Wigner 3D indexing of :math:`(\ell,m)` to 1D index.
   * - :func:`~hp_ang2pix`
     - Convert angles to HEALPix index for HEALPix ring ordering scheme.
   * - :func:`~hp_getidx`
     - Compute HEALPix harmonic index.
   * - :func:`~lm2lm_hp`
     - Converts from 1D indexed harmonic coefficients to HEALPix (healpy) indexed coefficients.

.. list-table:: Sampling & dimensionality conversions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~flm_2d_to_1d`
     - Convert from 2D indexed harmonic coefficients to 1D indexed coefficients.
   * - :func:`~flm_1d_to_2d`
     - Convert from 1D indexed harmnonic coefficients to 2D indexed coefficients.
   * - :func:`~flm_hp_to_2d`
     - Converts from HEALPix (healpy) indexed harmonic coefficients to 2D indexed coefficients.
   * - :func:`~flm_2d_to_hp`
     - Converts from 2D indexed harmonic coefficients to HEALPix (healpy) indexed coefficients.
   * - :func:`~flm_3d_to_1d`
     - Convert from 3D indexed Wigner coefficients to 1D indexed coefficients.
   * - :func:`~flm_1d_to_3d`
     - Convert from 1D indexed Wigner coefficients to 3D indexed coefficients.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Sampling Functions

   spherical_samples
   wigner_samples
