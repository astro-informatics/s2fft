from jax import config

config.update("jax_enable_x64", True)
import pytest
import numpy as np

from s2fft.transforms import wigner
from s2fft.base_transforms import wigner as base_wigner
from s2fft.recursions.price_mcewen import (
    generate_precomputes_wigner,
    generate_precomputes_wigner_jax,
)

L_to_test = [6, 7]
N_to_test = [2]
L_lower_to_test = [0, 2]
sampling_to_test = ["mw", "mwss", "dh"]
method_to_test = ["numpy", "jax"]
reality_to_test = [False, True]
multiple_gpus = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inverse_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
):
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f_check = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)

    if method.lower() == "jax":
        precomps = generate_precomputes_wigner_jax(
            L, N, sampling, None, False, reality, L_lower
        )
    else:
        precomps = generate_precomputes_wigner(
            L, N, sampling, None, False, reality, L_lower
        )
    f = wigner.inverse(
        flmn, L, N, None, sampling, method, reality, precomps, spmd, L_lower
    )
    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_forward_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
):
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)

    if method.lower() == "jax":
        precomps = generate_precomputes_wigner_jax(
            L, N, sampling, None, True, reality, L_lower
        )
    else:
        precomps = generate_precomputes_wigner(
            L, N, sampling, None, True, reality, L_lower
        )
    flmn_check = wigner.forward(
        f, L, N, None, sampling, method, reality, precomps, spmd, L_lower
    )
    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)
