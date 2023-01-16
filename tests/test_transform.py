import healpy as hp
import numpy as np
import pyssht as ssht
import pytest
from jax.config import config

import s2fft as s2f

config.update("jax_enable_x64", True)

L_to_test = [6, 7, 8]
L_lower_to_test = [0, 1, 2]
spin_to_test = [-2, -1, 0, 1, 2]
nside_to_test = [2, 4, 8]
L_to_nside_ratio = [2, 3]
sampling_to_test = ["mw", "mwss", "dh"]
method_to_test = [
    "direct",
    "sov",
    "sov_fft",
    "sov_fft_vectorized",
]
method_to_test_forward_only = [
    "jax_vmap_double",
    "jax_vmap_scan",
    "jax_vmap_loop",
    "jax_map_double",
    "jax_map_scan",
]
reality_to_test = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_inverse(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
):
    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f_check = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    f = s2f.transform._inverse(
        flm, L, spin, sampling, method, L_lower=L_lower, reality=reality
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_inverse_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    method: str,
    reality: bool,
):
    sampling = "healpix"
    L = ratio * nside
    flm = flm_generator(L=L, reality=True)
    flm_hp = s2f.samples.flm_2d_to_hp(flm, L)
    f_check = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    f = s2f.transform._inverse(
        flm, L, 0, sampling, method=method, nside=nside, reality=reality
    )

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test + method_to_test_forward_only)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_forward(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
):
    flm = flm_generator(L=L, spin=spin, reality=reality, L_lower=L_lower)

    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    flm_recov = s2f.transform._forward(
        f, L, spin, sampling, method, L_lower=L_lower, reality=reality
    )

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("method", method_to_test + method_to_test_forward_only)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_forward_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    method: str,
    reality: bool,
):
    sampling = "healpix"
    L = ratio * nside
    flm = flm_generator(L=L, reality=True)  # should this be reality=reality?

    if method in ["direct", "sov", "sov_fft", "sov_fft_vectorized"]:
        f = s2f.transform._inverse(
            flm, L, sampling=sampling, method=method, nside=nside
        )
    # use 'direct' for JAX approaches
    else:
        f = s2f.transform._inverse(
            flm, L, sampling=sampling, method="direct", nside=nside
        )

    flm_direct = s2f.transform._forward(
        f, L, sampling=sampling, method=method, nside=nside, reality=reality
    )
    flm_direct_hp = s2f.samples.flm_2d_to_hp(flm_direct, L)

    flm_check = hp.sphtfunc.map2alm(np.real(f), lmax=L - 1, iter=0)

    np.testing.assert_allclose(flm_direct_hp, flm_check, atol=1e-14)



@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_nside_to_L_exceptions(flm_generator, nside: int):
    sampling = "healpix"
    spin = 0
    L = 2 * nside - 1

    flm = flm_generator(L=L, reality=True)
    f = s2f.transform._inverse(flm, L, spin, sampling, "direct", nside)

    with pytest.raises(AssertionError) as e:
        s2f.transform.forward(f, L, spin, sampling, nside)

    with pytest.raises(AssertionError) as e:
        s2f.transform.inverse(flm, L, spin, sampling, nside)


@pytest.mark.parametrize("L", L_to_test)
def test_L_lower_exception(flm_generator, L: int):
    spin = 0
    sampling = "mw"

    flm = flm_generator(L=L, reality=False)
    f = s2f.transform.inverse(flm, L, spin, sampling)

    with pytest.raises(AssertionError) as e:
        s2f.transform.forward(f, L, spin, sampling, L_lower=-1)

    with pytest.raises(AssertionError) as e:
        s2f.transform.forward(f, L, spin, sampling, L_lower=L)

    with pytest.raises(AssertionError) as e:
        s2f.transform.inverse(flm, L, spin, sampling, L_lower=-1)

    with pytest.raises(AssertionError) as e:
        s2f.transform.inverse(flm, L, spin, sampling, L_lower=L)
