import pytest
import numpy as np
import pyssht as ssht
import healpy as hp
from s2fft.sampling import s2_samples as samples


nside_to_test = [16, 32]


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_samples_n_and_angles(L: int, sampling: str):
    # Test ntheta and nphi
    ntheta = samples.ntheta(L, sampling)
    nphi = samples.nphi_equiang(L, sampling)
    (ntheta_ssht, nphi_ssht) = ssht.sample_shape(L, sampling.upper())
    assert (ntheta, nphi) == pytest.approx((ntheta_ssht, nphi_ssht))

    # Test thetas and phis
    t = np.arange(0, ntheta)
    thetas = samples.t2theta(t, L, sampling)
    p = np.arange(0, nphi)
    phis = samples.p2phi_equiang(L, p, sampling)
    thetas_ssht, phis_ssht = ssht.sample_positions(L, sampling.upper())
    np.testing.assert_allclose(thetas, thetas_ssht, atol=1e-14)
    np.testing.assert_allclose(phis, phis_ssht, atol=1e-14)

    # Test direct thetas and phis
    np.testing.assert_allclose(samples.thetas(L, sampling), thetas_ssht, atol=1e-14)
    np.testing.assert_allclose(samples.phis_equiang(L, sampling), phis_ssht, atol=1e-14)


@pytest.mark.parametrize("ind", [15, 16])
def test_samples_index_conversion(ind: int):
    (el, m) = samples.ind2elm(ind)

    ind_check = samples.elm2ind(el, m)

    assert ind == ind_check


@pytest.mark.parametrize("L", [15, 16])
def test_samples_ncoeff(L: int):
    n = 0
    for el in range(0, L):
        for m in range(-el, el + 1):
            n += 1

    assert samples.ncoeff(L) == pytest.approx(n)


@pytest.mark.parametrize("nside", nside_to_test)
def test_samples_n_and_angles_hp(nside: int):
    ntheta = samples.ntheta(L=0, sampling="healpix", nside=nside)
    assert ntheta == 4 * nside - 1

    npix = hp.nside2npix(nside)
    hp_angles = np.zeros((npix, 2))
    for i in range(npix):
        hp_angles[i] = hp.pix2ang(nside, i)

    s2f_hp_angles = np.zeros((npix, 2))
    thetas = samples.thetas(L=0, sampling="healpix", nside=nside)
    entry = 0
    for ring in range(ntheta):
        phis = samples.phis_ring(ring, nside)
        s2f_hp_angles[entry : entry + len(phis), 0] = thetas[ring]
        s2f_hp_angles[entry : entry + len(phis), 1] = phis
        entry += len(phis)

    np.testing.assert_allclose(s2f_hp_angles, hp_angles, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
def test_hp_ang2pix(nside: int):
    for i in range(12 * nside**2):
        theta, phi = hp.pix2ang(nside, i)
        j = samples.hp_ang2pix(nside, theta, phi)
        assert i == j


def test_samples_exceptions():
    L = 10

    with pytest.raises(ValueError) as e:
        samples.phis_equiang(L, sampling="healpix")

    with pytest.raises(ValueError) as e:
        samples.phis_equiang(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        samples.ntheta(L, sampling="healpix")

    with pytest.raises(ValueError) as e:
        samples.ntheta(sampling="mw")

    with pytest.raises(ValueError) as e:
        samples.ntheta(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        samples.ntheta_extension(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        samples.nphi_equiang(L, sampling="healpix")

    with pytest.raises(ValueError) as e:
        samples.nphi_equiang(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        samples.nphi_ring(-1, nside=2)

    with pytest.raises(ValueError) as e:
        samples.t2theta(t=0, L=L, sampling="healpix")

    with pytest.raises(ValueError) as e:
        samples.t2theta(t=0, L=L, sampling="foo")

    with pytest.raises(ValueError) as e:
        samples.t2theta(t=0, sampling="mw")
