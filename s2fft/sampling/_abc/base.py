from abc import ABC, abstractmethod

import numpy as np


class Samples(ABC):
    r"""
    Abstract API structure that all sampling schemes must adhere to.

    All sampling schemes are required to provide methods for computing the
    :math:`(\theta, \phi)` coordinates that the samples will be placed at.
    They must additionally provide properties that specify the shape of the
    arrays that will be used to store the harmonic coefficients and signal
    values on the sphere during computations.

    Providing a value for the attribute `N` allows the instance to be used
    for SO3 sampling as well as S2 sampling.
    """

    N: int | None
    L: int

    @property
    def _n_gamma(self) -> int:
        r"""Number of :math:`\gamma` samples when sampling :math:`SO(3)`."""
        return 2 * self.N - 1

    @abstractmethod
    @property
    def n_theta(self) -> int:
        r"""Number of :math:`\theta` samples for sampling scheme at specified resolution."""

    @abstractmethod
    @property
    def thetas(self) -> np.ndarray:
        r"""Compute :math:`\theta` samples for given sampling scheme."""

    @abstractmethod
    @property
    def f_shape(self) -> tuple[int, int]:
        """Shape of spherical signal."""

    @property
    def f_shape_so3(self) -> tuple[int, int, int]:
        r"""Pixel-space sampling shape for signal on :math:`SO(3)`."""
        return self._n_gamma, *reversed(self.f_shape)

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

    @abstractmethod
    @property
    def ftm_shape(self) -> tuple[int, int]:
        """Shape of intermediate array, before/after latitudinal step."""

    def __init__(self, L: int, N: int | None = None):
        """
        Initialise the sampling scheme.

        Args:
            L (int): Harmonic band-limit the sampling will use.

            N (int, optional): Parameter `N` for SO3 sampling. If not provided, sample scheme
                can only be used on S2.

        """
        self.L = L
        self.N = N

    @abstractmethod
    def n_phi(self, theta_index: int) -> int:
        r"""Number of :math:`\phi` samples for given sampling scheme, on the given ring."""

    @abstractmethod
    def phis(self, theta_index: int) -> np.ndarray:
        r"""Compute :math:`\phi` samples for given sampling scheme, on the given ring."""

    def so3_ready(self, *, throw_on_false: bool = False) -> bool:
        r"""
        Whether this instance can be used to sample :math:`SO(3)` (`True`) or not (`False`).

        Sampling schemes must be provided the `N` parameter in order to be used
        in :math:`SO(3)` transforms. Without this parameter, the instance can
        only be used to sample from :math:`\mathbb{S}^2`.

        Args:
            throw_on_false (bool): If `True`, an exception is raised if the sampling scheme cannot
                be used to sample from :math:`SO(3)`, rather than simply returning `False`. Intended
                for use in a fail-fast check at the beginning of a computation.

        Raises:
            RuntimeError: If the scheme cannot sample :math:`SO(3)`, and the method has been
                provided `throw_on_false = True`.

        Returns:
            bool: `True` if the scheme can sample :math:`SO(3)`, otherwise `False`.

        """
        N_not_set = self.N is None
        if throw_on_false and N_not_set:
            raise RuntimeError(
                "Parameter N not set for sampling scheme, cannot sample from SO(3)"
            )

        return not N_not_set
