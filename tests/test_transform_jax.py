from jax import config

config.update("jax_enable_x64", True)
import pytest
import pyssht as ssht
import numpy as np
import healpy as hp

from s2fft import samples
from s2fft.jax_transforms import transform
from s2fft.wigner.price_mcewen import generate_precomputes

L_to_test = [6, 7, 8]
spin_to_test = [-2, -1, 0, 1, 2]
nside_to_test = [2, 4, 8]
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
    f = transform.inverse(
        flm,
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
    )
    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse_healpix(
    flm_generator,
    nside: int,
    method: str,
    reality: bool,
):
    sampling = "healpix"
    L = 2 * nside
    flm = flm_generator(L=L, reality=True)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f_check = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    precomps = generate_precomputes(L, 0, sampling, nside)
    f = transform.inverse(
        flm,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
    )

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm = flm_generator(L=L, spin=spin, reality=reality)

    f = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    precomps = generate_precomputes(L, spin, sampling, None, True)
    flm_check = transform.forward(
        f,
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
    )

    np.testing.assert_allclose(flm, flm_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward_healpix(
    flm_generator,
    nside: int,
    method: str,
    reality: bool,
):
    sampling = "healpix"
    L = 2 * nside
    flm = flm_generator(L=L, reality=True)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    precomps = generate_precomputes(L, 0, sampling, nside, True)
    flm_check = transform.forward(
        f,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
    )
    flm_check = samples.flm_2d_to_hp(flm_check, L)

    flm = hp.sphtfunc.map2alm(f, lmax=L - 1, iter=0)

    np.testing.assert_allclose(flm, flm_check, atol=1e-14)
