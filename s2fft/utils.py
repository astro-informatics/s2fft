import numpy as np
import s2fft as s2f
from s2fft.sampling import elm2ind


def generate_flm(L, spin=0, reality=False):
    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)

    if reality == False:
        flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)
        # For spin signals all flms for el < spin = 0, therfore first spin**2 coefficients = 0
        flm[: spin**2] = 0.0
        return flm
    else:
        for el in range(spin, L):
            flm[elm2ind(el, 0)] = np.random.rand()
            for em in range(1, el + 1):
                flm[elm2ind(el, em)] = np.random.rand() + 1j * np.random.rand()
                flm[elm2ind(el, -em)] = -(1 ** (em)) * np.conj(
                    flm[elm2ind(el, em)]
                )
        return flm
