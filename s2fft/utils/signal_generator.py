from __future__ import annotations

import numpy as np
import torch

from s2fft.sampling import s2_samples as samples
from s2fft.sampling import so3_samples as wigner_samples


def generate_flm(
    rng: np.random.Generator,
    L: int,
    L_lower: int = 0,
    spin: int = 0,
    reality: bool = False,
    using_torch: bool = False,
) -> np.ndarray | torch.Tensor:
    r"""
    Generate a 2D set of random harmonic coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        rng (Generator): Random number generator.

        L (int): Harmonic band-limit.

        L_lower (int, optional): Harmonic lower bound. Defaults to 0.

        spin (int, optional): Harmonic spin. Defaults to 0.

        reality (bool, optional): Reality of signal. Defaults to False.

        using_torch (bool, optional): Desired frontend functionality. Defaults to False.

    Returns:
        np.ndarray: Random set of spherical harmonic coefficients.

    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    min_el = max(L_lower, abs(spin))
    flm[min_el:L, L - 1] = rng.standard_normal(L - min_el)
    if not reality:
        flm[min_el:L, L - 1] += 1j * rng.standard_normal(L - min_el)
    m_indices, el_indices = np.triu_indices(n=L, k=1, m=L) + np.array([[1], [0]])
    if min_el > 0:
        in_range_el = el_indices >= min_el
        m_indices = m_indices[in_range_el]
        el_indices = el_indices[in_range_el]
    len_indices = len(m_indices)
    flm[el_indices, L - 1 - m_indices] = rng.standard_normal(
        len_indices
    ) + 1j * rng.standard_normal(len_indices)
    flm[el_indices, L - 1 + m_indices] = (-1) ** m_indices * np.conj(
        flm[el_indices, L - 1 - m_indices]
    )

    return torch.from_numpy(flm) if using_torch else flm


def generate_flmn(
    rng: np.random.Generator,
    L: int,
    N: int = 1,
    L_lower: int = 0,
    reality: bool = False,
    using_torch: bool = False,
) -> np.ndarray | torch.Tensor:
    r"""
    Generate a 3D set of random Wigner coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        rng (Generator): Random number generator.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations
            (i.e. directionality). Defaults to 1.

        L_lower (int, optional): Harmonic lower bound. Defaults to 0.

        reality (bool, optional): Reality of signal. Defaults to False.

        using_torch (bool, optional): Desired frontend functionality. Defaults to False.

    Returns:
        np.ndarray: Random set of Wigner coefficients.

    """
    flmn = np.zeros(wigner_samples.flmn_shape(L, N), dtype=np.complex128)

    for n in range(-N + 1, N):
        for el in range(max(L_lower, abs(n)), L):
            if reality:
                flmn[N - 1 + n, el, 0 + L - 1] = rng.normal()
                flmn[N - 1 - n, el, 0 + L - 1] = (-1) ** n * flmn[
                    N - 1 + n,
                    el,
                    0 + L - 1,
                ]
            else:
                flmn[N - 1 + n, el, 0 + L - 1] = rng.normal() + 1j * rng.normal()

            for m in range(1, el + 1):
                flmn[N - 1 + n, el, m + L - 1] = rng.normal() + 1j * rng.normal()
                if reality:
                    flmn[N - 1 - n, el, -m + L - 1] = (-1) ** (m + n) * np.conj(
                        flmn[N - 1 + n, el, m + L - 1]
                    )
                else:
                    flmn[N - 1 + n, el, -m + L - 1] = rng.normal() + 1j * rng.normal()

    return torch.from_numpy(flmn) if using_torch else flmn
