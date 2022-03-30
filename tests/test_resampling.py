import pytest
import numpy as np
import s2fft as s2f

from .utils import *


def test_periodic_extension_invalid_sampling():

    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError) as e:
        s2f.resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="healpix")

    with pytest.raises(ValueError) as e:
        s2f.resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="dh")


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize("spin", [0])
@pytest.mark.parametrize("reality", [False])
def test_periodic_extension_mwss(flm_generator, L: int, spin: int, reality: bool):

    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling="mwss")

    f_ext = s2f.resampling.periodic_extension(f, L, spin, sampling="mwss")

    f_ext_check = s2f.resampling.periodic_extension_spatial_mwss(f, L, spin)

    np.testing.assert_allclose(f_ext, f_ext_check, atol=1e-10)
