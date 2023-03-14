from jax import config

config.update("jax_enable_x64", True)
import pytest
import pyssht as ssht
import numpy as np
import healpy as hp

from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical
from s2fft.recursions.price_mcewen import generate_precomputes

L_to_test = [6, 7]
L_lower_to_test = [0, 2]
spin_to_test = [-2, 0, 1]
nside_to_test = [4, 5]
sampling_to_test = ["mw", "mwss", "dh"]
method_to_test = ["numpy", "jax"]
reality_to_test = [False, True]
multiple_gpus = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f_check = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=reality,
    )

    precomps = generate_precomputes(L, spin, sampling, L_lower=L_lower)
    f = spherical.inverse(
        flm,
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse_healpix(
    flm_generator,
    nside: int,
    method: str,
    spmd: bool,
):
    sampling = "healpix"
    L = 2 * nside
    flm = flm_generator(L=L, spin=0, reality=True)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f_check = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    precomps = generate_precomputes(L, 0, sampling, nside, False)
    f = spherical.inverse(
        flm,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
    )

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)

    f = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=reality,
    )

    precomps = generate_precomputes(L, spin, sampling, None, True, L_lower)
    flm_check = spherical.forward(
        f,
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(flm, flm_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward_healpix(
    flm_generator,
    nside: int,
    method: str,
    spmd: bool,
):
    sampling = "healpix"
    L = 2 * nside
    flm = flm_generator(L=L, spin=0, reality=True)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    precomps = generate_precomputes(L, 0, sampling, nside, True)
    flm_check = spherical.forward(
        f,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
    )
    flm_check = samples.flm_2d_to_hp(flm_check, L)

    flm = hp.sphtfunc.map2alm(f, lmax=L - 1, iter=0)

    np.testing.assert_allclose(flm, flm_check, atol=1e-14)
