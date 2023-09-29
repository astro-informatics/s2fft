import pytest
import numpy as np
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature
from s2fft.base_transforms import spherical


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
def test_quadrature_mw_weights(flm_generator, L: int, sampling: str):
    spin = 0

    q = quadrature.quad_weights(L, sampling, spin)

    flm = flm_generator(L, spin, reality=False)

    f = spherical.inverse(flm, L, spin, sampling)

    integral = flm[0, 0 + L - 1] * np.sqrt(4 * np.pi)
    q = np.reshape(q, (-1, 1))

    nphi = samples.nphi_equiang(L, sampling)
    Q = q.dot(np.ones((1, nphi)))

    integral_check = np.sum(Q * f)

    np.testing.assert_allclose(integral, integral_check, atol=1e-14)


def test_quadrature_exceptions():
    L = 10

    with pytest.raises(ValueError) as e:
        quadrature.quad_weights_transform(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        quadrature.quad_weights(L, sampling="foo")
