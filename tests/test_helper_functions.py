import pytest 
from pytest import fixture
import numpy as np 
import pyssht as ssht 
import s2fft as s2f


@fixture
def signal_generator():
    def generate_complex_ssht(L,  Method="MW", Spin=0, Reality=False):
        ncoeff = s2f.sampling.ncoeff(L)
        flm = np.zeros(ncoeff, dtype=np.complex128)
        flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)
        f = ssht.inverse(flm, L, Method=Method, Spin=Spin, Reality=False)
        return f, flm
    return generate_complex_ssht

