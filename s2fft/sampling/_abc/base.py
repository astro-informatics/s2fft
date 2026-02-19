from abc import ABC, abstractmethod

import numpy as np


class Samples(ABC):
    """Abstract API structure that all sampling schemes must adhere to."""

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

    @abstractmethod
    @property
    def ftm_shape(self) -> tuple[int, int]:
        """Shape of intermediate array, before/after latitudinal step."""
