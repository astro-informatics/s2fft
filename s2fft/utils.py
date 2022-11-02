import numpy as np
import s2fft.samples as samples


def generate_flm(
    rng: np.random.Generator, L: int, spin: int = 0, reality: bool = False
) -> np.ndarray:
    r"""Generate a 2D set of random harmonic coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        rng (Generator): Random number generator.

        L (int): Harmonic band-limit.

        spin (int, optional): Harmonic spin. Defaults to 0.

        reality (bool, optional): Reality of signal. Defaults to False.

    Returns:
        np.ndarray: Random set of spherical harmonic coefficients.

    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    for el in range(spin, L):

        if reality:
            flm[el, 0 + L - 1] = rng.uniform()
        else:
            flm[el, 0 + L - 1] = rng.uniform() + 1j * rng.uniform()

        for m in range(1, el + 1):
            flm[el, m + L - 1] = rng.uniform() + 1j * rng.uniform()
            if reality:
                flm[el, -m + L - 1] = (-1) ** m * np.conj(flm[el, m + L - 1])
            else:
                flm[el, -m + L - 1] = rng.uniform() + 1j * rng.uniform()

    return flm
