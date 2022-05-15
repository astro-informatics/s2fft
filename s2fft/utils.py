import numpy as np
import s2fft.samples as samples


def generate_flm(L: int, spin: int = 0, reality: bool = False) -> np.ndarray:
    r"""Generate a 2D set of random harmonic coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        L (int): Harmonic bandlimit.

        spin (int): Angular spin.

        reality (bool): Reality of signal (0 = Complex (default), 1 = Real).

    Returns:
        np.ndarray: Random set of harmonic coefficients.

    """
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)

    for el in range(spin, L):

        if reality:
            flm[el, 0 + L - 1] = np.random.rand()
        else:
            flm[el, 0 + L - 1] = np.random.rand() + 1j * np.random.rand()

        for m in range(1, el + 1):
            flm[el, m + L - 1] = np.random.rand() + 1j * np.random.rand()
            if reality:
                flm[el, -m + L - 1] = (-1) ** m * np.conj(flm[el, m + L - 1])
            else:
                flm[el, -m + L - 1] = np.random.rand() + 1j * np.random.rand()

    return flm
