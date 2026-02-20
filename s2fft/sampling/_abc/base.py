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

    @abstractmethod
    @property
    def ftm_shape(self) -> tuple[int, int]:
        """Shape of intermediate array, before/after latitudinal step."""

    def __init__(self, N: int | None = None):
        """
        Initialise the sampling scheme.

        Args:
            N (int, optional): Parameter `N` for SO3 sampling. If not provided, sample scheme
                can only be used on S2.

        """
        self.N = N

    @abstractmethod
    def n_phi(self, theta_index: int) -> int:
        r"""Number of :math:`\phi` samples for given sampling scheme, on the given ring."""

    @abstractmethod
    def phis(self, theta_index: int) -> np.ndarray:
        r"""Compute :math:`\phi` samples for given sampling scheme, on the given ring."""
