from abc import abstractmethod

import numpy as np

from .base import Samples


class PhiEquiangularSamples(Samples):
    r"""
    Mixin for :math:`\phi`-equiangular sampling schemes.

    Class provides the `phis` property, assuming equiangular sampling in that
    coordinate. Indexes for the :math:`\phi` samples are generated using `numpy.arange`,
    and then passed to the abstract method `_phi_index_to_value` which converts the
    indexes to the actual :math:`\phi` values.
    """

    @abstractmethod
    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        r"""
        Convert index to :math:`\phi` angle for sampling scheme.

        Args:
            p (int): :math:`\phi` index.

        Returns:
            float: :math:`\phi` sample(s) for given sampling scheme.

        """

    @property
    def phis(self) -> np.ndarray:
        return self._phi_index_to_value(np.arange(0, self.n_phi))
