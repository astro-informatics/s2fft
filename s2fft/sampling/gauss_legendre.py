import numpy as np
from typing_extensions import override

from ._abc.l_based import BandwidthSamples
from ._abc.phi_equiangular import PhiEquiangularSamples


class GaussLegendre(BandwidthSamples, PhiEquiangularSamples):
    """Gauss-Legendre sampling scheme."""

    @override
    @property
    def n_theta(self) -> int:
        return self.L

    @override
    @property
    def n_phi(self) -> int:
        return 2 * self.L - 1

    @override
    @property
    def thetas(self):
        return np.flip(np.arccos(np.polynomial.legendre.leggauss(self.n_theta)[0]))
