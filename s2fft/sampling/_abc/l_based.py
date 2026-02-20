from typing_extensions import override

from .base import Samples


class BandwidthSamples(Samples):
    """
    Abstract (sub)class for harmonic bandwidth-based sampling.

    These sampling methods depend on a harmonic bandwidth :math:`L` being provided
    at instantiation.
    """

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

    @override
    @property
    def f_shape(self) -> tuple[int, int]:
        return self.n_theta, self.n_phi

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

    @override
    @property
    def ftm_shape(self) -> tuple[int, int]:
        return self.n_theta, self.n_phi

    def __init__(self, L: int):
        """Init."""
        self.L = L
