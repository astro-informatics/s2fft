from .s2_base import S2Samples


class BandLimitedSamples(S2Samples):
    """
    Abstract (sub)class for band-limited sampling.

    These sampling methods depend on a harmonic band-limit :math:`L` being provided
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

    def __init__(self, L: int) -> None:
        r"""
        Initialise sampling with harmonic band-limit `L`.

        Args:
            L (int): Harmonic band-limit.

        """
        self.L = L
