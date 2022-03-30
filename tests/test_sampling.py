import pytest
import numpy as np
import s2fft as s2f
import pyssht as ssht
import healpy as hp 


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_sampling_n_and_angles(L: int, sampling: str):

    # Test ntheta and nphi
    ntheta = s2f.sampling.ntheta(L, sampling)
    nphi = s2f.sampling.nphi_equiang(L, sampling)
    (ntheta_ssht, nphi_ssht) = ssht.sample_shape(L, sampling.upper())
    assert (ntheta, nphi) == pytest.approx((ntheta_ssht, nphi_ssht))

    # Test thetas and phis
    t = np.arange(0, ntheta)
    thetas = s2f.sampling.t2theta(L, t, sampling)
    p = np.arange(0, nphi)
    phis = s2f.sampling.p2phi_equiang(L, p, sampling)
    thetas_ssht, phis_ssht = ssht.sample_positions(L, sampling.upper())
    np.testing.assert_allclose(thetas, thetas_ssht, atol=1e-14)
    np.testing.assert_allclose(phis, phis_ssht, atol=1e-14)

    # Test direct thetas and phis
    np.testing.assert_allclose(
        s2f.sampling.thetas(L, sampling), thetas_ssht, atol=1e-14
    )
    np.testing.assert_allclose(
        s2f.sampling.phis_equiang(L, sampling), phis_ssht, atol=1e-14
    )


@pytest.mark.parametrize("ind", [15, 16])
def test_sampling_index_conversion(ind: int):

    (el, m) = s2f.sampling.ind2elm(ind)

    ind_check = s2f.sampling.elm2ind(el, m)

    assert ind == ind_check


@pytest.mark.parametrize("L", [15, 16])
def test_sampling_ncoeff(L: int):

    n = 0
    for el in range(0, L):
        for m in range(-el, el + 1):
            n += 1

    assert s2f.sampling.ncoeff(L) == pytest.approx(n)

def test_sampling_exception():

     L = 10

     with pytest.raises(ValueError) as e:
         s2f.sampling.phis_equiang(L, sampling="healpix")


@pytest.mark.parametrize("nside", [32, 64, 128])
def test_sampling_n_and_angles_hp(nside: int):

    ntheta = s2f.sampling.ntheta(L=0, sampling="healpix", nside=nside)
    assert(ntheta == 4*nside-1)

    npix = hp.nside2npix(nside)
    hp_angles = np.zeros((npix, 2))
    for i in range(npix):
        hp_angles[i] = hp.pix2ang(nside,i)
    
    s2f_hp_angles = np.zeros((npix, 2))
    thetas = s2f.sampling.thetas(L=0, sampling="healpix", nside=nside)
    entry = 0
    for ring in range(ntheta):
        phis = s2f.sampling.phis_ring(ring, nside)
        s2f_hp_angles[entry:entry+len(phis), 0] = thetas[ring]
        s2f_hp_angles[entry:entry+len(phis), 1] = phis
        entry += len(phis)
 
    np.testing.assert_allclose(s2f_hp_angles, hp_angles, atol=1e-14)

  
@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
def test_sampling_mw_weights(L: int, sampling: str):

    # TODO: move this and potentially do better
    # np.random.seed(2)

    spin = 0

    q = s2f.sampling.quad_weights(L, sampling, spin)

    # Create bandlimited signal
    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    integral = flm[0] * np.sqrt(4 * np.pi)
    q = np.reshape(q, (-1, 1))

    nphi = s2f.sampling.nphi_equiang(L, sampling)
    Q = q.dot(np.ones((1, nphi)))

    integral_check = np.sum(Q * f)

    print(f"sampling = {sampling}")
    print(f"q = {q}")
    # print(f"Q = {Q}")
    # print(f"q.shape = {q.shape}")
    # print(f"Q.shape = {Q.shape}")

    print(f"integral = {integral}")
    print(f"integral_check = {integral_check}")

    np.testing.assert_allclose(integral, integral_check, atol=1e-14)

