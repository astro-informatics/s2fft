from .s2_base import S2Samples


class SO3Samples(S2Samples):
    """
    Abstract API structure that all SO3-sampling schemes must adhere to.

    Class instances are STATIC after instantiation, as per the recommendations
    in `JAX's documentation<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#strategy-2-marking-self-as-static>`_.
    The harmonic band-limit, `self.L` and parameter `self.N` are thus assumed static
    after instantiation. However, to be safe we will manually implement the methods for
    equality and hashing this class, as recommended by JAX.
    """

    _N: int

    @property
    def N(self) -> int:
        """SO3 sampling parameter."""
        return self._N

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
        self._N = N
        super().__init__(L=L)

    def __hash__(self) -> int:
        return hash((self.L, self.N))

    def __eq__(self, other: "SO3Samples") -> bool:
        return super().__eq__(other) and self.N == other.N
