from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from s2fft.sampling import reindex
from s2fft.sampling import s2_samples as samples

nside_to_test = [16, 32]


@pytest.fixture
def cached_healpy_angles_test_case(cached_test_case_wrapper: Callable) -> Callable:
    def generate_data(nside: int) -> dict[str, np.ndarray]:
        import healpy

        npix = healpy.nside2npix(nside)
        hp_angles = np.zeros((npix, 2))
        for i in range(npix):
            hp_angles[i] = healpy.pix2ang(nside, i)

        return {"hp_angles": hp_angles}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.fixture
def cached_ssht_samples_test_case(cached_test_case_wrapper: Callable) -> Callable:
    def generate_data(L: int, sampling: str) -> dict[str, np.ndarray]:
        import pyssht

        ntheta, nphi = pyssht.sample_shape(L, sampling.upper())
        thetas, phis = pyssht.sample_positions(L, sampling.upper())

        return {"ntheta": ntheta, "nphi": nphi, "thetas": thetas, "phis": phis}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.mark.parametrize("L", [15, 16])
def test_fast_reindexing_functions(L: int):
    flm = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)
    flm_jax = jnp.array(flm)

    flm_1d = samples.flm_2d_to_1d(flm, L)
    flm_1d_jax = reindex.flm_2d_to_1d_fast(flm_jax, L)
    np.testing.assert_allclose(flm_1d, flm_1d_jax)

    flm_2d = samples.flm_1d_to_2d(flm_1d, L)
    flm_2d_jax = reindex.flm_1d_to_2d_fast(flm_1d_jax, L)
    np.testing.assert_allclose(flm_2d, flm_2d_jax)

    flm_hp = samples.flm_2d_to_hp(flm_2d, L)
    flm_hp_jax = reindex.flm_2d_to_hp_fast(flm_2d_jax, L)
    np.testing.assert_allclose(flm_hp, flm_hp_jax)

    flm_2d = samples.flm_hp_to_2d(flm_hp, L)
    flm_2d_jax = reindex.flm_hp_to_2d_fast(flm_hp_jax, L)
    np.testing.assert_allclose(flm_2d, flm_2d_jax)


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl"])
def test_samples_n_and_angles(
    cached_ssht_samples_test_case: Callable, L: int, sampling: str
):
    # Test ntheta and nphi
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    ssht_data = cached_ssht_samples_test_case(L, sampling)
    assert (ntheta, nphi) == (ssht_data["ntheta"], ssht_data["nphi"])

    # Test thetas and phis
    if sampling.lower() == "gl":
        thetas = samples.thetas(L, sampling)
    else:
        t = np.arange(0, ntheta)
        thetas = samples.t2theta(t, L, sampling)
    p = np.arange(0, nphi)
    phis = samples.p2phi_equiang(L, p, sampling)
    np.testing.assert_allclose(thetas, ssht_data["thetas"], atol=1e-14)
    np.testing.assert_allclose(phis, ssht_data["phis"], atol=1e-14)

    # Test direct thetas and phis
    np.testing.assert_allclose(
        samples.thetas(L, sampling), ssht_data["thetas"], atol=1e-14
    )
    np.testing.assert_allclose(
        samples.phis_equiang(L, sampling), ssht_data["phis"], atol=1e-14
    )


@pytest.mark.parametrize("ind", [15, 16])
def test_samples_index_conversion(ind: int):
    (el, m) = samples.ind2elm(ind)

    ind_check = samples.elm2ind(el, m)

    assert ind == ind_check


@pytest.mark.parametrize("L", [15, 16])
def test_samples_ncoeff(L: int):
    n = 0
    for el in range(0, L):
        for _ in range(-el, el + 1):
            n += 1

    assert samples.ncoeff(L) == pytest.approx(n)


@pytest.mark.parametrize("nside", nside_to_test)
def test_samples_n_and_angles_hp(cached_healpy_angles_test_case: Callable, nside: int):
    ntheta = samples.ntheta(L=0, sampling="healpix", nside=nside)
    assert ntheta == 4 * nside - 1

    healpy_data = cached_healpy_angles_test_case(nside)

    s2f_hp_angles = np.zeros((12 * nside**2, 2))
    thetas = samples.thetas(L=0, sampling="healpix", nside=nside)
    entry = 0
    for ring in range(ntheta):
        phis = samples.phis_ring(ring, nside)
        s2f_hp_angles[entry : entry + len(phis), 0] = thetas[ring]
        s2f_hp_angles[entry : entry + len(phis), 1] = phis
        entry += len(phis)

    np.testing.assert_allclose(s2f_hp_angles, healpy_data["hp_angles"], atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
def test_hp_ang2pix(cached_healpy_angles_test_case: Callable, nside: int):
    healpy_data = cached_healpy_angles_test_case(nside)
    for i in range(12 * nside**2):
        theta, phi = healpy_data["hp_angles"][i]
        j = samples.hp_ang2pix(nside, theta, phi)
        assert i == j


def test_samples_exceptions():
    L = 10

    with pytest.raises(ValueError):
        samples.phis_equiang(L, sampling="healpix")

    with pytest.raises(ValueError):
        samples.phis_equiang(L, sampling="foo")

    with pytest.raises(ValueError):
        samples.ntheta(L, sampling="healpix")

    with pytest.raises(ValueError):
        samples.ntheta(sampling="mw")

    with pytest.raises(ValueError):
        samples.ntheta(L, sampling="foo")

    with pytest.raises(ValueError):
        samples.ntheta_extension(L, sampling="foo")

    with pytest.raises(ValueError):
        samples.nphi_equiang(L, sampling="healpix")

    with pytest.raises(ValueError):
        samples.nphi_equiang(L, sampling="foo")

    with pytest.raises(ValueError):
        samples.nphi_ring(-1, nside=2)

    with pytest.raises(ValueError):
        samples.t2theta(t=0, L=L, sampling="healpix")

    with pytest.raises(ValueError):
        samples.t2theta(t=0, L=L, sampling="foo")

    with pytest.raises(ValueError):
        samples.t2theta(t=0, sampling="mw")
