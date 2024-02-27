from jax import config

config.update("jax_enable_x64", True)
import pytest
import numpy as np
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch
from s2fft.base_transforms import spherical


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
def test_quadrature_mw_weights(flm_generator, L: int, sampling: str, method: str):
    spin = 0

    if method.lower() == "numpy":
        q = quadrature.quad_weights(L, sampling, spin)
    elif method.lower() == "jax":
        q = quadrature_jax.quad_weights(L, sampling)
    elif method.lower() == "torch":
        q = quadrature_torch.quad_weights(L, sampling).numpy()

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
