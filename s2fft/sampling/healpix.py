import numpy as np
from typing_extensions import override

from ._abc.thetas_from_index import ThetasFromIndex


class HEALPix(ThetasFromIndex):
    """HEALPix sampling scheme."""

    n_side: int

    @property
    def n_phi_equatorial_band(self) -> int:
        r"""Number of :math:`\phi` samples within the equatorial band for HEALPix sampling scheme."""
        return 4 * self.n_side

    @override
    @property
    def n_theta(self) -> int:
        return 4 * self.n_side - 1

    @override
    @property
    def f_shape(self) -> tuple[int]:
        return (12 * self.n_side**2,)

    @override
    @property
    def ftm_shape(self) -> tuple[int, int]:
        # TODO: This is technically self.n_theta, 2L for some input L - it's just that the default shape is set to 2L = 4nside. Note however that dependence on L means this needs to be a function, not a method...
        return self.n_theta, self.n_phi_equatorial_band

    def __init__(self, n_side: int, N: int | None = None) -> None:
        r"""
        Initialise HEALPix sampling scheme with :math:`N_{side}` pixels.

        Args:
            n_side (int): Number of pixels, :math:`N_{side}`.

            N (int, optional): Parameter `N` for SO3 sampling. If not provided, sample scheme
                can only be used on S2.

        """
        self.n_side = n_side
        super().__init__(N=N)

    def _phi_index_to_value_on_ring(
        self, theta_index: int, phi_index: np.ndarray
    ) -> np.ndarray:
        r"""
        Convert index to :math:`\phi` angle for HEALPix for given :math:`\theta` ring.

        Args:
            theta_index (int): :math:`\theta` index of ring.

            phi_index (np.ndarray): :math:`\phi` indexes within ring.

        Returns:
            np.ndarray: :math:`\phi` angle.

        """
        shift = 1 / 2
        if (theta_index + 1 >= self.n_side) & (theta_index + 1 <= 3 * self.n_side):
            shift *= (theta_index - self.n_side + 2) % 2
            factor = np.pi / (2 * self.n_side)
            return factor * (phi_index + shift)
        elif theta_index + 1 > 3 * self.n_side:
            factor = np.pi / (2 * (4 * self.n_side - theta_index - 1))
        else:
            factor = np.pi / (2 * (theta_index + 1))
        return factor * (phi_index + shift)

    @override
    def _theta_index_to_value(self, theta_index: np.ndarray) -> np.ndarray:
        r"""
        Convert (ring) index to :math:`\theta` angle for HEALPix sampling scheme.

        Args:
            theta_index (np.ndarray): :math:`\theta` indexes.

        Returns:
            np.ndarray: :math:`\theta` angle(s) for passed HEALPix (ring) index or indices.

        """
        z = np.zeros_like(theta_index)
        z[theta_index < self.n_side - 1] = 1 - (
            theta_index[theta_index < self.n_side - 1] + 1
        ) ** 2 / (3 * self.n_side**2)

        z[(theta_index >= self.n_side - 1) & (theta_index <= 3 * self.n_side - 1)] = (
            4 / 3
            - 2
            * (
                theta_index[
                    (theta_index >= self.n_side - 1)
                    & (theta_index <= 3 * self.n_side - 1)
                ]
                + 1
            )
            / (3 * self.n_side)
        )

        z[
            (theta_index > 3 * self.n_side - 1) & (theta_index <= 4 * self.n_side - 2)
        ] = (
            4 * self.n_side
            - 1
            - theta_index[
                (theta_index > 3 * self.n_side - 1)
                & (theta_index <= 4 * self.n_side - 2)
            ]
        ) ** 2 / (3 * self.n_side**2) - 1

        return np.arccos(z)

    def _zphi_to_pixel(self, z: float, phi: float) -> int:
        r"""
        Convert angles to HEALPix index for HEALPix ring ordering scheme, using
        :math:`z=\cos(\theta)`.

        Note:
            Translated function from HEALPix Java implementation.

        Args:
            z (float): Cosine of spherical :math:`\theta` angle, i.e. :math:`\cos(\theta)`.

            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.

        """
        tt = 2 * phi / np.pi
        za = np.abs(z)
        nl2 = int(2 * self.n_side)
        nl4 = int(4 * self.n_side)
        ncap = int(nl2 * (self.n_side - 1))
        npix = int(12 * self.n_side**2)
        if za < 2 / 3:  # equatorial region
            jp = int(self.n_side * (0.5 + tt - 0.75 * z))
            jm = int(self.n_side * (0.5 + tt + 0.75 * z))

            ir = int(self.n_side + 1 + jp - jm)
            kshift = 0
            if ir % 2 == 0:
                kshift = 1
            ip = int((jp + jm - self.n_side + kshift + 1) / 2) + 1
            ipix1 = ncap + nl4 * (ir - 1) + ip

        else:  # North and South polar caps
            tp = tt - int(tt)
            tmp = np.sqrt(3.0 * (1.0 - za))
            jp = int(self.n_side * tp * tmp)
            jm = int(self.n_side * (1.0 - tp) * tmp)

            ir = jp + jm + 1
            ip = int(tt * ir) + 1
            if ip > 4 * ir:
                ip = ip - 4 * ir

            ipix1 = 2 * ir * (ir - 1) + ip
            if z <= 0.0:
                ipix1 = npix - 2 * ir * (ir + 1) + ip

        return ipix1 - 1

    def angle_to_pixel(self, theta: float, phi: float) -> int:
        r"""
        Convert angles to HEALPix index for HEALPix ring ordering scheme.

        Args:
            theta (float): Spherical :math:`\theta` angle.

            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.

        """
        return self._zphi_to_pixel(np.cos(theta), phi)

    @override
    def n_phi(self, theta_index: int) -> int:
        r"""
        Number of :math:`\phi` samples for HEALPix sampling on given :math:`\theta`
        ring.

        Args:
            theta_index (int): Index of HEALPix :math:`\theta` ring.

        Raises:
            ValueError: Invalid ring index given.

        Returns:
            int: Number of :math:`\phi` samples on given :math:`\theta` ring.

        """
        if (theta_index >= 0) and (theta_index < self.n_side - 1):
            return 4 * (theta_index + 1)

        elif (theta_index >= self.n_side - 1) and (theta_index <= 3 * self.n_side - 1):
            return 4 * self.n_side

        elif (theta_index > 3 * self.n_side - 1) and (
            theta_index <= 4 * self.n_side - 2
        ):
            return 4 * (4 * self.n_side - theta_index - 1)

        else:
            raise ValueError(
                f"Ring t={theta_index} not contained by nside={self.n_side}"
            )

    @override
    def phis(self, theta_index: int) -> np.ndarray:
        r"""
        Compute :math:`\phi` samples for given :math:`\theta` HEALPix ring.

        Args:
            theta_index (int): :math:`\theta` index.

        Returns:
            np.ndarray: :math:`\phi` angles.

        """
        p = np.arange(0, self.n_phi_ring(theta_index)).astype(np.float64)
        return self._phi_index_to_value_on_ring(theta_index, p)

    def ring_phase_shift(
        self,
        L: int,
        theta_index: int,
        forward: bool = False,
        reality: bool = False,
    ) -> np.ndarray:
        r"""
        Generates a phase shift vector for HEALPix for a given :math:`\theta` ring.

        Args:
            L (int): Harmonic band-limit.

            theta_index (int): :math:`\theta` index of ring.

            forward (bool, optional): Whether to provide forward or inverse shift.
                Defaults to False.

            reality (bool, optional): Whether the signal on the sphere is real.  If so,
                conjugate symmetry is exploited to reduce computational costs.
                Defaults to False.

        Returns:
            np.ndarray: Vector of phase shifts with shape :math:`[2L-1]`.

        """
        phi_offset = self._phi_index_to_value_on_ring(theta_index, 0)
        sign = -1 if forward else 1
        m_start_ind = 0 if reality else -L + 1
        return np.exp(sign * 1j * np.arange(m_start_ind, L) * phi_offset)
