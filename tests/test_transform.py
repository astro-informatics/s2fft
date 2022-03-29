import pytest
import numpy as np
import s2fft as s2f
import pyssht as ssht


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_direct(L: int, spin: int, sampling: str):

    ncoeff = s2f.sampling.ncoeff(L)

    flm = np.zeros((ncoeff, 1), dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    f_check = ssht.inverse(flm, L, Method=sampling.upper(), Spin=spin, Reality=False)

    np.testing.assert_allclose(f, f_check, atol=1e-14)
