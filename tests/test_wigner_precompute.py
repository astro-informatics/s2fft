import pytest
import numpy as np
import s2fft
from s2fft.precompute_transforms.wigner import inverse, forward
from s2fft.precompute_transforms.construct import wigner_kernel
from s2fft.base_transforms import wigner as base

L_to_test = [8, 10]
N_to_test = [2, 3]
nside_to_test = [4, 6]
L_to_nside_ratio = [2]
reality_to_test = [False, True]
sampling_schemes = ["mw", "mwss", "dh"]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
):
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality)

    kernel = wigner_kernel(L, N, reality, sampling, forward=False)
    f_check = inverse(flmn, L, N, kernel, sampling, reality, method)

    np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
):
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, sampling=sampling, reality=reality)
    flmn = base.forward(f, L, N, sampling=sampling, reality=reality)

    kernel = wigner_kernel(L, N, reality, sampling, forward=True)
    flmn_check = forward(f, L, N, kernel, sampling, reality, method)
    np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_wigner_transform_healpix(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality, nside)

    kernel = wigner_kernel(L, N, reality, sampling, nside=nside, forward=False)
    f_check = inverse(flmn, L, N, kernel, sampling, reality, method, nside)

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_wigner_transform_healpix(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality, nside)
    flmn_check = base.forward(f, L, N, 0, sampling, reality, nside)

    kernel = wigner_kernel(L, N, reality, sampling, nside=nside, forward=True)
    flmn_forward = forward(f, L, N, kernel, sampling, reality, method, nside)
    np.testing.assert_allclose(flmn_forward, flmn_check, atol=1e-5, rtol=1e-5)
