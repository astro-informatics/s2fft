import numpy as np
import pytest
from jax import config

from s2fft.base_transforms import spherical
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl", "cc"])
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
def test_quadrature_mw_weights(flm_generator, L: int, sampling: str, method: str):
    spin = 0

    if sampling == "cc" and method != "numpy" :
        pytest.skip("cc to be implemented in jax and pytorch")
    # 
   
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

    print(q.shape)
    print(Q.shape)
    print(f.shape)
    integral_check = np.sum(Q * f)

    np.testing.assert_allclose(integral, integral_check, atol=1e-14)


def test_quadrature_exceptions():
    L = 10

    with pytest.raises(ValueError):
        quadrature.quad_weights_transform(L, sampling="foo")

    with pytest.raises(ValueError):
        quadrature.quad_weights(L, sampling="foo")
