from .base import Samples


class BandLimitedSamples(Samples):
    """
    Abstract (sub)class for band-limited sampling.

    These sampling methods depend on a harmonic band-limit :math:`L` being provided
    at instantiation.
    """

    L: int

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

    @property
    def flmn_shape(L: int, N: int) -> tuple[int, int, int]:
        r"""Shape of Wigner coefficients for a signal on :math:`SO(3)`."""
        return 2 * N - 1, L, 2 * L - 1

    def __init__(self, L: int, N: int | None = None) -> None:
        r"""
        Initialise sampling with harmonic band-limit `L`.

        Args:
            L (int): Harmonic band-limit.

            N (int, optional): Parameter `N` for SO3 sampling. If not provided, sample scheme
                can only be used on S2.

        """
        self.L = L
        super().__init__(N=N)
