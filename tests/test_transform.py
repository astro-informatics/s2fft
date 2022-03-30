import pytest
import numpy as np
import s2fft as s2f
import pyssht as ssht

from .utils import *

# @pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_direct(
    flm_generator, L: int, spin: int, sampling: str
):

    flm = flm_generator(L=L, spin=spin, reality=False)
    f_check = ssht.inverse(flm, L, Method=sampling.upper(), Spin=spin, Reality=False)
    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("sampling", ["dh"])
def test_transform_forward_direct(
    flm_generator, L: int, spin: int, sampling: str
):

    # TODO: move this and potentially do better
    np.random.seed(2)

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    flm_recov = s2f.transform.forward_direct(f, L, spin, sampling)

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)
