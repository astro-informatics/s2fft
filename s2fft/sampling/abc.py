from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override


class Samples(ABC):
    """Samples."""

    L: int

    @property
    def n_coeff(self) -> int:
        """
        Number of spherical harmonic coefficients for given band-limit L.

        Args:
            L (int, optional): Harmonic band-limit.

        Returns:
            int: Number of spherical harmonic coefficients.

        """
        return self.elm2ind(self.L - 1, self.L - 1) + 1

    def __init__(self, L: int):
        """Init."""
        self.L = L

    @abstractmethod
    @property
    def n_phi(self) -> int:
        r"""Number of :math:`\phi` samples for given sampling scheme."""

    @abstractmethod
    @property
    def n_theta(self) -> int:
        r"""Number of :math:`\theta` samples for sampling scheme at specified resolution."""

    @abstractmethod
    @property
    def thetas(self) -> np.ndarray:
        r"""Compute :math:`\theta` samples for given sampling scheme."""

    @abstractmethod
    @property
    def phis(self) -> np.ndarray:
        r"""Compute :math:`\phi` samples for given sampling scheme."""

    @abstractmethod
    @property
    def f_shape(self) -> tuple[int, int]:
        """Shape of spherical signal."""
        # TODO: This is identical for all non-HEALPIX schemes... some refactoring?
        # Possibly separate schemes into HP and non-HP :joy:

    @abstractmethod
    @property
    def ftm_shape(self) -> tuple[int, int]:
        """Shape of intermediate array, before/after latitudinal step."""

    @property
    def flm_shape(self) -> tuple[int, int]:
        r"""
        Standard shape of harmonic coefficients.

        Args:
            L (int, optional): Harmonic band-limit.

        Returns:
            Tuple[int]: Sampling array shape, with indexing :math:`[\ell, m]`.

        """
        return self.L, 2 * self.L - 1

    @staticmethod
    def elm2ind(el: int, m: int) -> int:
        r"""
        Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.

        1D index is defined by `el**2 + el + m`.

        Warning:
            Note that 1D storage of spherical harmonic coefficients is *not* the default.

        Args:
            el (int): Harmonic degree :math:`\ell`.

            m (int): Harmonic order :math:`m`.

        Returns:
            int: Corresponding 1D index value.

        """
        return el**2 + el + m

    @staticmethod
    def ind2elm(ind: int) -> tuple:
        r"""
        Convert from 1D spherical harmonic index to 2D index of :math:`(\ell,m)`.

        Warning:
            Note that 1D storage of spherical harmonic coefficients is *not* the default.

        Args:
            ind (int): 1D spherical harmonic index.

        Returns:
            tuple: `(el,m)` defining spherical harmonic degree and order.

        """
        el = np.floor(np.sqrt(ind))

        m = ind - el**2 - el

        return el, m


class PhiEquiangularSamples(Samples):
    r""":math:`\phi`-equiangular sampling schemes."""

    @abstractmethod
    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        r"""
        Convert index to :math:`\phi` angle for sampling scheme.

        Args:
            p (int): :math:`\phi` index.

        Returns:
            float: :math:`\phi` sample(s) for given sampling scheme.

        """

    @override
    @property
    def phis(self) -> np.ndarray:
        return self._phi_index_to_value(np.arange(0, self.n_phi))
