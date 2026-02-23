from .s2_base import S2Samples


class SO3Samples(S2Samples):
    """Abstract API structure that all SO3-sampling schemes must adhere to."""

    N: int

    @property
    def _n_gamma(self) -> int:
        r"""Number of :math:`\gamma` samples when sampling :math:`SO(3)`."""
        return 2 * self.N - 1

    @property
    def f_shape_so3(self) -> tuple[int, int, int]:
        r"""Pixel-space sampling shape for signal on :math:`SO(3)`."""
        return self._n_gamma, *reversed(self.f_shape)

    @property
    def flmn_shape(self) -> tuple[int, int, int]:
        r"""Shape of Wigner coefficients for a signal on :math:`SO(3)`."""
        return 2 * self.N - 1, self.L, 2 * self.L - 1

    @property
    def fnab_shape(self) -> tuple[int, int, int]:
        r"""
        Shape of Wigner space sampling of rotation group :math:`SO(3)`.

        For sampling schemes that possess a sampling theorem (e.g., MW, MWSS, GL, DH)
        this usually identical in shape to `self.f_shape_so3`. For schemes that lack
        such a theorem (e.g., HEALPix) it may be of a different shape.
        TODO: Don't think this is the actual reason, more likely to do with the sample
        placements themselves. Check with the others. Possibly make this abstract to
        enforce explicit overwrite in concrete classes / create a mixin for certain
        groups of schemes.

        The default implementation returns `self.f_shape_so3`. Subclasses should
        overwrite as necessary.
        """
        return self.f_shape_so3

    def __init__(self, L: int, N: int):
        """
        Initialise the sampling scheme.

        Args:
            L (int): Harmonic band-limit the sampling will use.

            N (int, optional): Parameter `N` for SO3 sampling.

        """
        self.N = N
        super().__init__(L=L)
