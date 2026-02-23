import numpy as np
from typing_extensions import override

from ._abc.phi_equiangular import PhiEquiangularSamples
from ._abc.so3_base import SO3Samples


class GaussLegendre(PhiEquiangularSamples):
    """Gauss-Legendre sampling scheme."""

    @override
    @property
    def _n_phi(self) -> int:
        return 2 * self.L - 1

    @override
    @property
    def n_theta(self) -> int:
        return self.L

    @override
    @property
    def thetas(self):
        return np.flip(np.arccos(np.polynomial.legendre.leggauss(self.n_theta)[0]))


class GaussLegendreSO3(SO3Samples, GaussLegendre):
    """Gauss-Legendre sampling scheme on :math:`SO(3)`."""
