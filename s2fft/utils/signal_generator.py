from __future__ import annotations

import numpy as np

from s2fft.sampling import s2_samples as samples
from s2fft.sampling import so3_samples as wigner_samples


def complex_normal(
    rng: np.random.Generator,
    size: int | tuple[int],
    var: float,
) -> np.ndarray:
    """
    Generate array of samples from zero-mean complex normal distribution.

    For `z ~ ComplexNormal(0, var)` we have that `imag(z) ~ Normal(0, var/2)` and
    `real(z) ~ Normal(0, var/2)` where `Normal(μ, σ²)` is the (real-valued) normal
    distribution with mean parameter `μ` and variance parameter `σ²`.

    Args:
        rng: Numpy random generator object to generate samples using.
        size: Output shape of array to generate.
        var: Variance of complex normal distribution to generate samples from.

    Returns:
        Complex-valued array of shape `size` contained generated samples.

    """
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) * (
        var / 2
    ) ** 0.5


def complex_el_and_m_indices(L: int, min_el: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of el, m indices for accessing complex harmonic coefficients.

    Equivalent to nested list-comprehension based implementation

    .. code-block:: python

        el_indices, m_indices = np.array(
            [(el, m) for el in range(min_el, L) for m in range(1, el + 1)]
        ).T

    For `L, min_el = 1024, 0`, this implementation is around 80x quicker in
    benchmarks compared to list-comprehension implementation.

    Args:
        L: Harmonic band-limit.
        min_el: Inclusive lower-bound for el indices.

    Returns:
        Tuple `(el_indices, m_indices)` with both entries 1D integer-valued NumPy arrays
        of same size, with values of corresponding entries corresponding to pairs of
        el and m indices.

    """
    el_indices, m_indices = np.tril_indices(m=L, k=-1, n=L)
    m_indices += 1
    if min_el > 0:
        in_range_el = el_indices >= min_el
        el_indices = el_indices[in_range_el]
        m_indices = m_indices[in_range_el]
    return el_indices, m_indices


def generate_flm(
    rng: np.random.Generator,
    L: int,
    L_lower: int = 0,
    spin: int = 0,
    reality: bool = False,
) -> np.ndarray:
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

    Returns:
        np.ndarray: Random set of spherical harmonic coefficients.

    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    min_el = max(L_lower, abs(spin))
    # m = 0 coefficients are always real
    flm[min_el:L, L - 1] = rng.standard_normal(L - min_el)
    # Construct arrays of m and el indices for entries in flm corresponding to complex-
    # valued coefficients (m > 0)
    el_indices, m_indices = complex_el_and_m_indices(L, min_el)
    len_indices = len(m_indices)
    # Generate independent complex coefficients for positive m
    flm[el_indices, L - 1 + m_indices] = complex_normal(rng, len_indices, var=2)
    if reality:
        # Real-valued signal so set complex coefficients for negative m using conjugate
        # symmetry such that flm[el, L - 1 - m] = (-1)**m * flm[el, L - 1 + m].conj
        flm[el_indices, L - 1 - m_indices] = (-1) ** m_indices * (
            flm[el_indices, L - 1 + m_indices].conj()
        )
    else:
        # Non-real signal so generate independent complex coefficients for negative m
        flm[el_indices, L - 1 - m_indices] = complex_normal(rng, len_indices, var=2)
    return flm


def generate_flmn(
    rng: np.random.Generator,
    L: int,
    N: int = 1,
    L_lower: int = 0,
    reality: bool = False,
) -> np.ndarray:
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

    Returns:
        np.ndarray: Random set of Wigner coefficients.

    """
    flmn = np.zeros(wigner_samples.flmn_shape(L, N), dtype=np.complex128)
    for n in range(-N + 1, N):
        min_el = max(L_lower, abs(n))
        # Separately deal with m = 0 case
        if reality:
            if n == 0:
                # For m = n = 0
                # flmn[N - 1, el, L - 1] = flmn[N - 1, el, L - 1].conj (real-valued)
                # Generate independent real coefficients for n = 0
                flmn[N - 1, min_el:L, L - 1] = rng.standard_normal(L - min_el)
            elif n > 0:
                # Generate independent complex coefficients for positive n
                flmn[N - 1 + n, min_el:L, L - 1] = complex_normal(
                    rng, L - min_el, var=2
                )
                # For m = 0, n > 0
                # flmn[N - 1 - n, el, L - 1] = (-1)**n * flmn[N - 1 + n, el, L - 1].conj
                flmn[N - 1 - n, min_el:L, L - 1] = (-1) ** n * (
                    flmn[N - 1 + n, min_el:L, L - 1].conj()
                )
        else:
            flmn[N - 1 + n, min_el:L, L - 1] = complex_normal(rng, L - min_el, var=2)
        # Construct arrays of m and el indices for entries in flmn slices for n
        # corresponding to complex-valued coefficients (m > 0)
        el_indices, m_indices = complex_el_and_m_indices(L, min_el)
        len_indices = len(m_indices)
        # Generate independent complex coefficients for positive m
        flmn[N - 1 + n, el_indices, L - 1 + m_indices] = complex_normal(
            rng, len_indices, var=2
        )
        if reality:
            # Real-valued signal so set complex coefficients for negative m using
            # conjugate symmetry relationship
            #     flmn[N - 1 - n, el, L - 1 - m] =
            #         (-1)**(m + n) * flmn[N - 1 + n, el, L - 1 + m].conj
            # As (m_indices + n) can be negative use floating point value (-1.0) as
            # base of exponentation operation to avoid Numpy
            # 'ValueError: Integers to negative integer powers are not allowed' error
            flmn[N - 1 - n, el_indices, L - 1 - m_indices] = (-1.0) ** (
                m_indices + n
            ) * flmn[N - 1 + n, el_indices, L - 1 + m_indices].conj()
        else:
            # Complex signal so generate independent complex coefficients for negative m
            flmn[N - 1 + n, el_indices, L - 1 - m_indices] = complex_normal(
                rng, len_indices, var=2
            )
    return flmn
