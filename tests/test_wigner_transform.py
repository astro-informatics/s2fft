import jax
import jax.numpy as jnp
import numpy as np
import pytest

from s2fft.base_transforms import wigner as base_wigner
from s2fft.recursions.price_mcewen import (
    generate_precomputes_wigner,
    generate_precomputes_wigner_jax,
)
from s2fft.transforms import wigner

jax.config.update("jax_enable_x64", True)

L_to_test = [6, 7]
N_to_test = [2]
L_lower_to_test = [0, 2]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
method_to_test = ["numpy", "jax", "torch"]
reality_to_test = [False, True]

_generate_precomputes_functions = {
    "jax": generate_precomputes_wigner_jax,
    "numpy": generate_precomputes_wigner,
    # torch method wraps jax so use jax to generate precomputess
    "torch": generate_precomputes_wigner_jax,
}


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inverse_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    method: str,
    reality: bool,
):
    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f_check = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)
    precomps = _generate_precomputes_functions[method](
        L, N, sampling, None, False, reality, L_lower
    )
    f = wigner.inverse(flmn, L, N, None, sampling, method, reality, precomps, L_lower)
    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_forward_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    method: str,
    reality: bool,
):
    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)
    precomps = _generate_precomputes_functions[method](
        L, N, sampling, None, True, reality, L_lower
    )
    flmn_check = wigner.forward(
        f, L, N, None, sampling, method, reality, precomps, L_lower
    )
    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ssht_c_backend_inverse_wigner_transform(
    flmn_generator, L: int, N: int, L_lower: int, sampling: str, reality: bool
):
    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f_check = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)

    flmn = jnp.array(flmn)
    f = wigner.inverse(flmn, L, N, None, sampling, "jax_ssht", reality, L_lower=L_lower)
    np.testing.assert_allclose(f, f_check, atol=1e-12)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ssht_c_backend_forward_wigner_transform(
    flmn_generator, L: int, N: int, L_lower: int, sampling: str, reality: bool
):
    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f = base_wigner.inverse(flmn, L, N, L_lower, sampling, reality)

    flmn_check = wigner.forward(
        f, L, N, None, sampling, "jax_ssht", reality, L_lower=L_lower
    )
    np.testing.assert_allclose(flmn, flmn_check, atol=1e-12)


def test_N_exceptions(flmn_generator):
    N = 10
    L = 16
    flmn = flmn_generator(L=L, N=N)
    f = base_wigner.inverse(flmn, L, N)

    with pytest.raises(Warning):
        wigner.inverse(flmn, L, N)

    with pytest.raises(Warning):
        wigner.forward(f, L, N)


def test_sampling_ssht_backend_exceptions(flmn_generator):
    L = 16
    N = 1
    flmn = flmn_generator(L=L, N=N)
    f = base_wigner.inverse(flmn, L, N)

    with pytest.raises(ValueError):
        wigner.inverse(flmn, L, N, sampling="healpix", method="jax_ssht")

    with pytest.raises(ValueError):
        wigner.forward(f, L, N, sampling="healpix", method="jax_ssht")
