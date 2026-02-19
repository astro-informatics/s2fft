import numpy as np
from typing_extensions import override

from .abc import PhiEquiangularSamples


class GLSamples(PhiEquiangularSamples):
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
    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        return 2 * phi_index * np.pi / (2 * self.L - 1)

    @override
    @property
    def ftm_shape(self) -> tuple[int, int]:
        return self.n_theta, 2 * self.L - 1

    @override
    @property
    def f_shape(self) -> tuple[int, int]:
        return self.n_theta, self.n_phi

    @override
    @property
    def thetas(self):
        return np.flip(np.arccos(np.polynomial.legendre.leggauss(self.n_theta)[0]))
