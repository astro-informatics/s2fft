import pytest
import numpy as np
import s2fft as s2f
from s2fft.experimental import transforms_healpix as s2fhp
import pyssht as ssht
import healpy as hp

from .utils import *


L_to_test = [3, 4, 5]
nside_to_test = [2, 4, 8]
spin_to_test = [0, 1, 2]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_direct(flm_generator, L: int, spin: int, sampling: str):

    flm = flm_generator(L=L, spin=spin, reality=False)
    f_check = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )
    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


# @pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("nside", nside_to_test)
def test_transform_inverse_direct_healpix(flm_generator, nside: int):
    L = 2 * nside
    flm = flm_generator(L=L, reality=True)
    flm_hp = s2f.samples.flm_2d_to_hp(flm, L)
    f_check = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    f = s2fhp.inverse_direct_healpix(flm, L, nside)

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_sov(flm_generator, L: int, spin: int, sampling: str):

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = s2f.transform.inverse_sov(flm, L, spin, sampling)

    f_check = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)


# @pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("nside", nside_to_test)
def test_transform_inverse_sov_healpix(flm_generator, nside: int):
    L = 2 * nside
    flm = flm_generator(L=L, reality=True)
    flm_hp = s2f.samples.flm_2d_to_hp(flm, L)
    f_check = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)

    f = s2fhp.inverse_sov_healpix(flm, L, nside)

    np.testing.assert_allclose(np.real(f), np.real(f_check), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_sov_fft(flm_generator, L: int, spin: int, sampling: str):

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling)

    f_check = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_sov_fft_vect(
    flm_generator, L: int, spin: int, sampling: str
):

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = s2f.transform.inverse_sov_fft_vect(flm, L, spin, sampling)

    f_check = s2f.transform.inverse_sov_fft(flm, L, spin, sampling)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["dh"])
def test_transform_forward_direct(flm_generator, L: int, spin: int, sampling: str):

    # TODO: move this and potentially do better
    np.random.seed(2)

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    flm_recov = s2f.transform.forward_direct(f, L, spin, sampling)

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)


# @pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", L_to_test)
def test_transform_forward_direct_healpix(flm_generator, L: int):
    nside = 2 * L
    flm = flm_generator(L=L, reality=True)
    f = s2fhp.inverse_direct_healpix(flm, L, nside)

    flm_direct = s2fhp.forward_direct_healpix(f, L, nside)
    flm_direct_hp = s2f.samples.flm_2d_to_hp(flm_direct, L)
    flm_check = hp.sphtfunc.map2alm(np.real(f), lmax=L - 1)

    np.testing.assert_allclose(flm_direct_hp, flm_check, atol=1e-2)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["dh"])
def test_transform_forward_sov(flm_generator, L: int, spin: int, sampling: str):

    # TODO: move this and potentially do better
    np.random.seed(2)

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    flm_recov = s2f.transform.forward_sov(f, L, spin, sampling)

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize("spin", [0, 1, 2])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_forward_sov_fft(flm_generator, L: int, spin: int, sampling: str):

    # TODO: move this and potentially do better
    np.random.seed(2)

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    flm_recov = s2f.transform.forward_sov_fft(f, L, spin, sampling)

    np.set_printoptions(linewidth=150, precision=4)
    print(f"f = {f}")
    print(f"flm = {flm}")
    print(f"flm_recov = {flm_recov}")

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)


# @pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", L_to_test)
def test_transform_forward_sov_healpix(flm_generator, L: int):
    nside = 2 * L
    flm = flm_generator(L=L, reality=True)
    f = s2fhp.inverse_sov_healpix(flm, L, nside)

    flm_direct = s2fhp.forward_sov_healpix(f, L, nside)
    flm_direct_hp = s2f.samples.flm_2d_to_hp(flm_direct, L)
    flm_check = hp.sphtfunc.map2alm(np.real(f), lmax=L - 1)

    np.testing.assert_allclose(flm_direct_hp, flm_check, atol=1e-2)
