import pytest
import numpy as np
import s2fft as s2f
from .utils import *


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
def test_quadrature_mw_weights(flm_generator, L: int, sampling: str):

    # TODO: move this and potentially do better
    # np.random.seed(2)

    spin = 0

    q = s2f.quadrature.quad_weights(L, sampling, spin)

    flm = flm_generator(L, spin, reality=False)

    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling)

    integral = flm[0, 0 + L - 1] * np.sqrt(4 * np.pi)
    q = np.reshape(q, (-1, 1))

    nphi = s2f.samples.nphi_equiang(L, sampling)
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


def test_quadrature_exceptions():

    L = 10

    with pytest.raises(ValueError) as e:
        s2f.quadrature.quad_weights_transform(L, sampling="foo")

    with pytest.raises(ValueError) as e:
        s2f.quadrature.quad_weights(L, sampling="foo")
