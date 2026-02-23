from abc import ABC, abstractmethod

import numpy as np


class S2Samples(ABC):
    r"""
    Abstract API structure that all S2-sampling schemes must adhere to.

    All sampling schemes are required to provide methods for computing the
    :math:`(\theta, \phi)` coordinates that the samples will be placed at.
    They must additionally provide properties that specify the shape of the
    arrays that will be used to store the harmonic coefficients and signal
    values on the sphere during computations.

    Class instances are STATIC after instantiation, as per the recommendations
    in `JAX's documentation<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#strategy-2-marking-self-as-static>`_.
    The only class attribute is the harmonic band-limit, `self.L` (which is
    actually a property to protect `self._L`, where the value is stored). However,
    to be safe we will manually implement the methods for equality and hashing this
    class, as recommended by JAX.
    """

    _L: int

    @property
    def L(self) -> int:
        """
        Harmonic band-limit used by the sampling scheme.

        `L` is static, to allow for JIT-compilation of class methods.
        """
        return self._L

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
    def flm_shape(self) -> tuple[int, int]:
        r"""
        Standard shape of harmonic coefficients.

        Args:
            L (int, optional): Harmonic band-limit.

        Returns:
            Tuple[int]: Sampling array shape, with indexing :math:`[\ell, m]`.

        """
        return self.L, 2 * self.L - 1

    @abstractmethod
    @property
    def ftm_shape(self) -> tuple[int, int]:
        """Shape of intermediate array, before/after latitudinal step."""

    def __init__(self, L: int):
        """
        Initialise the S2-sampling scheme.

        Args:
            L (int): Harmonic band-limit the sampling will use.

        """
        self._L = L

    def __hash__(self) -> int:
        return hash((self.L,))

    def __eq__(self, other: "S2Samples") -> bool:
        return isinstance(other, type(self)) and (self._L == other._L)

    @abstractmethod
    def n_phi(self, theta_index: int) -> int:
        r"""Number of :math:`\phi` samples for given sampling scheme, on the given ring."""

    @abstractmethod
    def phis(self, theta_index: int) -> np.ndarray:
        r"""Compute :math:`\phi` samples for given sampling scheme, on the given ring."""
