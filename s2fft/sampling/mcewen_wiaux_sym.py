import numpy as np
from typing_extensions import override

from ._abc.l_based import BandwidthSamples
from ._abc.phi_equiangular import PhiEquiangularSamples
from ._abc.thetas_from_index import ThetasFromIndex


class McEwenWiauxSymmetric(BandwidthSamples, PhiEquiangularSamples, ThetasFromIndex):
    """McEwen & Wiaux Symmetric sampling scheme."""

    @override
    @property
    def n_theta(self) -> int:
        return self.L + 1

    @override
    @property
    def n_phi(self) -> int:
        return 2 * self.L

    @override
    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        return 2 * phi_index * np.pi / self.n_phi

    @override
    def _theta_index_to_value(self, theta_index):
        return 2 * theta_index * np.pi / (2 * self.L)

    @override
    @property
    def ftm_shape(self) -> tuple[int, int]:
        return self.n_theta, self.n_phi
