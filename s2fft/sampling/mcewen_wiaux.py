import numpy as np
from typing_extensions import override

from ._abc.l_based import BandwidthSamples
from ._abc.phi_equiangular import PhiEquiangularSamples
from ._abc.thetas_from_index import ThetasFromIndex


class McEwenWiaux(BandwidthSamples, PhiEquiangularSamples, ThetasFromIndex):
    """McEwen & Wiaux sampling scheme."""

    @override
    @property
    def n_theta(self) -> int:
        return self.L

    @override
    @property
    def n_phi(self) -> int:
        return 2 * self.L - 1

    @override
    def _theta_index_to_value(self, theta_index):
        return (2 * theta_index + 1) * np.pi / (2 * self.L - 1)
