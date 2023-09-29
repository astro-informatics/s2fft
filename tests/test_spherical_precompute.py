import pytest
import numpy as np
from s2fft.precompute_transforms.spherical import inverse, forward
from s2fft.precompute_transforms.construct import spin_spherical_kernel
from s2fft.base_transforms import spherical as base

L_to_test = [6, 7]
spin_to_test = [-2, 0, 1]
nside_to_test = [4, 5]
L_to_nside_ratio = [2, 3]
sampling_to_test = ["mw", "mwss", "dh"]
reality_to_test = [True, False]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_inverse(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
):
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_check = base.inverse(flm, L, spin, sampling, reality=reality)

    kernel = spin_spherical_kernel(
        L,
        spin,
        reality,
        sampling,
        nside=None,
        forward=False,
    )
    f = inverse(flm, L, spin, kernel, sampling, reality, method)

    np.testing.assert_allclose(f, f_check, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_inverse_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flm = flm_generator(L=L, reality=True)
    f_check = base.inverse(flm, L, 0, sampling, nside, reality)

    kernel = spin_spherical_kernel(L, 0, reality, sampling, nside=nside, forward=False)
    f = inverse(flm, L, 0, kernel, sampling, reality, method, nside)

    np.testing.assert_allclose(f, f_check, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_forward(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
):
    flm = flm_generator(L=L, spin=spin, reality=reality)

    f = base.inverse(flm, L, spin, sampling, reality=reality)
    flm_check = base.forward(f, L, spin, sampling, reality=reality)

    kernel = spin_spherical_kernel(L, spin, reality, sampling, nside=None, forward=True)
    flm_recov = forward(f, L, spin, kernel, sampling, reality, method)
    for i in range(L):
        for j in range(2 * L - 1):
            print(flm_recov[i, j], flm_check[i, j])
    np.testing.assert_allclose(flm_check, flm_recov, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_forward_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flm = flm_generator(L=L, reality=True)
    f = base.inverse(flm, L, 0, sampling, nside, reality)
    flm_check = base.forward(f, L, 0, sampling, nside, reality)

    kernel = spin_spherical_kernel(L, 0, reality, sampling, nside=nside, forward=True)
    flm_recov = forward(f, L, 0, kernel, sampling, reality, method, nside)

    np.testing.assert_allclose(flm_recov, flm_check, atol=1e-12, rtol=1e-12)
