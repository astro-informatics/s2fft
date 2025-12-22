from collections.abc import Callable

import jax
import numpy as np
import pytest

from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms import fourier_wigner as fw
from s2fft.sampling import so3_samples as samples

jax.config.update("jax_enable_x64", True)

# Test cases
L_to_test = [16]
N_to_test = [2, 8, 16]
reality_to_test = [False, True]
sampling_schemes = ["mw", "mwss"]
methods_to_test = ["numpy", "jax"]
delta_method_to_test = ["otf", "precomp"]

# Test tolerance
atol = 1e-12


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("delta_method", delta_method_to_test)
def test_inverse_fourier_wigner_transform(
    cached_so3_test_case: Callable,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    delta_method: str,
):

    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    transform = fw.inverse_transform_jax if method == "jax" else fw.inverse_transform
    if delta_method.lower() == "precomp":
        precomps = (
            c.fourier_wigner_kernel_jax(L)
            if method == "jax"
            else c.fourier_wigner_kernel(L)
        )
    else:
        precomps = None
    f_check = transform(test_data["flmn"], L, N, precomps, reality, sampling)
    np.testing.assert_allclose(test_data["f_so3"], f_check.flatten("C"), atol=atol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("delta_method", delta_method_to_test)
def test_forward_fourier_wigner_transform(
    cached_so3_test_case: Callable,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    delta_method: str,
):

    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    f = test_data["f_so3"].reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )

    transform = fw.forward_transform_jax if method == "jax" else fw.forward_transform
    if delta_method.lower() == "precomp":
        precomps = (
            c.fourier_wigner_kernel_jax(L)
            if method == "jax"
            else c.fourier_wigner_kernel(L)
        )
    else:
        precomps = None
    flmn_check = transform(f, L, N, precomps, reality, sampling)
    np.testing.assert_allclose(test_data["flmn"], flmn_check, atol=atol)


@pytest.mark.parametrize("L", [8, 16, 32])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_fourier_wigner_transform_high_N(
    cached_so3_test_case: Callable, L: int, sampling: str, reality: bool
):
    N = L
    
    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    f = test_data["f_so3"].real if reality else test_data["f_so3"]
    precomps = c.fourier_wigner_kernel(L)
    f_check = fw.inverse_transform(test_data["flmn"], L, N, precomps, reality, sampling)

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=atol)


@pytest.mark.parametrize("L", [8, 16, 32])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_fourier_wigner_transform_high_N(
    cached_so3_test_case: Callable, L: int, sampling: str, reality: bool
):
    N = L

    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    f = test_data["f_so3"].reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )

    precomps = c.fourier_wigner_kernel_jax(L)
    flmn_check = fw.forward_transform_jax(f, L, N, precomps, reality, sampling)
    np.testing.assert_allclose(test_data["flmn_so3"], flmn_check, atol=atol)
