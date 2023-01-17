import pytest
import numpy as np
import s2fft
from s2fft.general_precompute.spin_spherical import inverse, forward
from s2fft.general_precompute.construct import spin_spherical_kernel

L_to_test = [6, 7, 8]
spin_to_test = [-2, -1, 0, 1, 2]
nside_to_test = [2, 4, 8]
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
    f_check = s2fft.transform.inverse(flm, L, spin, sampling, reality=reality)

    kernel = spin_spherical_kernel(
        L,
        spin,
        reality,
        sampling,
        nside=None,
        forward=False,
    )
    f = inverse(flm, L, spin, kernel, sampling, reality, method)

    np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


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
    f_check = s2fft.transform.inverse(flm, L, 0, sampling, nside, reality)

    kernel = spin_spherical_kernel(
        L, 0, reality, sampling, nside=nside, forward=False
    )
    f = inverse(flm, L, 0, kernel, sampling, reality, method, nside)

    np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


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

    f = s2fft.transform.inverse(flm, L, spin, sampling, reality=reality)
    flm_check = s2fft.transform.forward(f, L, spin, sampling, reality=reality)

    kernel = spin_spherical_kernel(
        L, spin, reality, sampling, nside=None, forward=True
    )
    flm_recov = forward(f, L, spin, kernel, sampling, reality, method)

    np.testing.assert_allclose(flm_check, flm_recov, atol=1e-5, rtol=1e-5)


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
    f = s2fft.transform.inverse(flm, L, 0, sampling, nside, reality)
    flm_check = s2fft.transform.forward(f, L, 0, sampling, nside, reality)

    kernel = spin_spherical_kernel(
        L, 0, reality, sampling, nside=nside, forward=True
    )
    flm_recov = forward(f, L, 0, kernel, sampling, reality, method, nside)

    np.testing.assert_allclose(flm_recov, flm_check, atol=1e-5, rtol=1e-5)
