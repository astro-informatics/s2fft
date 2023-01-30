import pytest
import numpy as np
from s2fft import samples
from s2fft.jax_transforms import transform
from s2fft.wigner.price_mcewen import generate_precomputes
import pyssht as ssht


L_to_test = [6, 7, 8]
L_lower_to_test = [0]
spin_to_test = [-2, -1, 0, 1, 2]
sampling_to_test = ["mw", "mwss", "dh"]
method_to_test = ["numpy", "jax"]
reality_to_test = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm = flm_generator(L=L, L_lower=0, spin=spin, reality=reality)
    f_check = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=reality,
    )

    precomps = generate_precomputes(L, spin, sampling)
    f = transform.inverse(flm, L, spin, sampling, method, reality, precomps)
    np.testing.assert_allclose(f, f_check, atol=1e-14)
