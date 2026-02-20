from abc import abstractmethod

import numpy as np
from typing_extensions import override

from .s2_base import S2Samples


class PhiEquiangularSamples(S2Samples):
    r"""
    Mixin for :math:`\phi`-equiangular sampling schemes.

    The number of :math:`\phi`-samples on each ring is constant for
    equiangular sampling schemes, and as such it is required to be provided
    as a property `self._n_phi` = :math:`N_{\phi}`. The public `self.n_phi`
    method ignores the `theta_index` argument and returns :math:`N_{\phi}`.

    Similarly, the `self.phis` method also ignores the `theta_index` argument,
    since the sampling scheme is equiangular in :math:`\phi`. As such, the
    `phis` method is implemented by the class, with the :math:`p`-th sample
    :math:`\phi_p  = \frac{2 p \pi}{N_{\phi}}`.

    Note that both the `f_shape` and `ftm_shape` properties will return a size
    of :math:`N_{\phi}` along the :math:`\phi`-dimension.
    """

    @abstractmethod
    @property
    def _n_phi(self) -> int:
        r"""Number of :math:`\phi` samples in each ring."""

    @override
    @property
    def f_shape(self) -> tuple[int, int]:
        return self.n_theta, self._n_phi

    @override
    @property
    def ftm_shape(self) -> tuple[int, int]:
        return self.n_theta, self._n_phi

    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        r"""
        Convert index to :math:`\phi` angle for equiangular sampling scheme.

        Args:
            phi_index (np.ndarray): :math:`\phi` index.

        Returns:
            np.ndarray: :math:`\phi` sample(s) for given sampling scheme.

        """
        return 2 * phi_index * np.pi / self._n_phi

    @override
    def n_phi(self, theta_index: int = None) -> int:
        r"""
        Number of :math:`\phi` samples for given sampling scheme, on the given ring.

        For :math:`\phi`-equiangular sampling schemes, the `theta_index` argument may
        be omitted (and is ignored by the method).
        """
        return self._n_phi

    @override
    def phis(self, theta_index: int = None) -> np.ndarray:
        r"""
        Compute :math:`\phi` samples for given sampling scheme, on the given ring.

        For :math:`\phi`-equiangular sampling schemes, the `theta_index` argument may
        be omitted (and is ignored by the method).
        """
        return self._phi_index_to_value(np.arange(0, self._n_phi))
