import pytest
import numpy as np
import s2fft as s2f
import pyssht as ssht


@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
def test_sampling_n_and_angles(L: int, sampling: str):

    # Test ntheta and nphi
    ntheta = s2f.sampling.ntheta(L, sampling)
    nphi = s2f.sampling.nphi_eqiang(L, sampling)
    (ntheta_ssht, nphi_ssht) = ssht.sample_shape(L, sampling.upper())
    assert (ntheta, nphi) == pytest.approx((ntheta_ssht, nphi_ssht))

    # Test thetas and phis
    t = np.arange(0, ntheta)
    thetas = s2f.sampling.t2theta(L, t, sampling)
    p = np.arange(0, nphi)
    phis = s2f.sampling.p2phi_eqiang(L, p, sampling)
    thetas_ssht, phis_ssht = ssht.sample_positions(L, sampling.upper())
    np.testing.assert_allclose(thetas, thetas_ssht, atol=1e-14)
    np.testing.assert_allclose(phis, phis_ssht, atol=1e-14)

    # Test direct thetas and phis
    np.testing.assert_allclose(
        s2f.sampling.thetas(L, sampling), thetas_ssht, atol=1e-14
    )
    np.testing.assert_allclose(
        s2f.sampling.phis_eqiang(L, sampling), phis_ssht, atol=1e-14
    )


def test_sampling_errors():

    L = 10

    with pytest.raises(NotImplementedError) as e:
        s2f.sampling.thetas(L, sampling="healpix")

    with pytest.raises(ValueError) as e:
        s2f.sampling.phis_eqiang(L, sampling="healpix")
