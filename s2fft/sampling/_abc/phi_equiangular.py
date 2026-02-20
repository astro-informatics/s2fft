import numpy as np

from .base import Samples


class PhiEquiangularSamples(Samples):
    r"""
    Mixin for :math:`\phi`-equiangular sampling schemes.

    Class provides the `phis` property, assuming equiangular sampling in that
    coordinate. Indexes for the :math:`\phi` samples are generated assuming
    equiangular sampling, so the :math:`p`-th coordinate :math:`\phi_p` is at

    $$ \phi_p = \frac{2 p \pi}{N_{\phi}}, $$

    where :math:`N_{\phi}` = `self.n_phi`.
    """

    def _phi_index_to_value(self, phi_index: np.ndarray) -> np.ndarray:
        r"""
        Convert index to :math:`\phi` angle for sampling scheme.

        Args:
            phi_index (np.ndarray): :math:`\phi` index.

        Returns:
            np.ndarray: :math:`\phi` sample(s) for given sampling scheme.

        """
        return 2 * phi_index * np.pi / self.n_phi

    @property
    def phis(self) -> np.ndarray:
        return self._phi_index_to_value(np.arange(0, self.n_phi))
