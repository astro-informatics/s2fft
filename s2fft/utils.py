import numpy as np
import pyssht as ssht
import s2fft as s2f


def generate_signal_ssht(L, Method="MW", Spin=0, Reality=False):
    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)

    if Reality == False:
        flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)
        flm[: Spin**2] = 0.0
        f = ssht.inverse(flm, L, Method=Method, Spin=Spin, Reality=False)
        return f, flm
    else:
        for el in range(Spin, L):
            flm[ssht.elm2ind(el, 0)] = np.random.rand() + 1j * np.random.rand()
            for em in range(1, el + 1):
                flm[ssht.elm2ind(el, em)] = np.random.rand() + 1j * np.random.rand()
                flm[ssht.elm2ind(el, -em)] = -(1 ** (em)) * np.conj(
                    flm[ssht.elm2ind(el, em)]
                )

        f = ssht.inverse(flm, L, Method=Method, Spin=Spin, Reality=True)
        return f, flm
