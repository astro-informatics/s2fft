import numpy as np
import pytest
from jax import config

from s2fft.base_transforms import spherical
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("L", [5, 6])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl", "cc", "f2"])
@pytest.mark.parametrize("method", ["numpy", "jax", "torch"])
def test_quadrature_weights_in_transform(
    flm_generator, L: int, sampling: str, method: str
):
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

    for sampling in ("cc", "f2"):
        with pytest.raises(ValueError, match="must be at least 1"):
            quadrature.quad_weights(0, sampling)


def _bandlimit_from_n_theta(n_theta, rule):
    # Compute bandlimit L such that ntheta(L, rule) in {n_theta, n_theta + 1}
    match rule:
        case "mw" | "gl":
            return n_theta
        case "f2" | "cc":
            return n_theta // 2 + 1
        case "mwss":
            return n_theta - 1
        case "dh":
            return (n_theta + 1) // 2


def _min_n_points_exact(rule, polynomial_degree):
    # Minimum number of quadrature points to exactly integrate polynomial of
    # specified degree for a given rule
    match rule:
        case "gl":
            return polynomial_degree // 2 + 1
        case "f2" | "cc":
            return polynomial_degree
        case "dh" | "mw":
            return polynomial_degree + 1
        case "mwss":
            return polynomial_degree + 2


def _min_n_points(rule):
    # Rules including both poles need a minimum of 2 points
    if rule in (
        samples.INCLUDES_NORTH_POLE_SCHEMES & samples.INCLUDES_SOUTH_POLE_SCHEMES
    ):
        return 2
    else:
        return 1


def check_quadrature_rule(f_and_integral, rule, n_points, quadrature_module, tol=1e-12):
    f, true_integral = f_and_integral
    L = _bandlimit_from_n_theta(n_points, rule)
    # As ntheta may be constrained to be an odd or even number given L,
    # we allow that ntheta may be equal to n_points or one more
    assert samples.ntheta(L, rule) in (n_points, n_points + 1)
    thetas = samples.thetas(L, sampling=rule)
    xs = np.cos(thetas)
    weights = {
        "cc": quadrature_module.quad_weights_cc_theta_only,
        "f2": quadrature_module.quad_weights_f2_theta_only,
        "mw": quadrature_module.quad_weights_mw_theta_only,
        "mwss": quadrature_module.quad_weights_mwss_theta_only,
        "gl": quadrature_module.quad_weights_gl_theta_only,
        "dh": quadrature_module.quad_weights_dh_theta_only,
    }[rule](L)
    quad_integral = (f(xs) * weights).sum()
    assert abs(quad_integral - true_integral) < tol


@pytest.mark.parametrize(
    "f_and_integral_and_degree",
    [
        (lambda x: 0.0, 0.0, 0),
        (lambda x: x + 1, 2.0, 1),
        (lambda x: x**2, 2 / 3, 2),
        (lambda x: x**3 - 2 * x**2 + x - 1, -10 / 3, 3),
        (lambda x: x**4 - x**2, -4 / 15, 4),
        (lambda x: x**8 - x**4 + x**3 - 2 * x**2 + x - 1, -158 / 45, 8),
    ],
)
@pytest.mark.parametrize("rule", ["cc", "f2", "mw", "mwss", "gl", "dh"])
@pytest.mark.parametrize("n_points_offset", [0, 1, 2, 4])
@pytest.mark.parametrize("quadrature_module", [quadrature, quadrature_jax])
def test_quadrature_polynomial(
    f_and_integral_and_degree, rule, n_points_offset, quadrature_module
):
    *f_and_integral, polynomial_degree = f_and_integral_and_degree
    # Number of quadrature points defined relative to minimum number
    # required to exactly integrate polynomial of specified degree
    # to ensure we test this minimum (when n_points_offset == 0)
    n_points = max(
        _min_n_points_exact(rule, polynomial_degree) + n_points_offset,
        _min_n_points(rule),
    )
    check_quadrature_rule(f_and_integral, rule, n_points, quadrature_module)


@pytest.mark.parametrize(
    "f_and_integral",
    [
        (lambda x: np.cos(x), np.sin(1) * 2),
        (lambda x: np.exp(x), np.exp(1) - np.exp(-1)),
        (lambda x: np.log(1 + x**2), 2 * np.log(2) - 4 + np.pi),
    ],
)
@pytest.mark.parametrize("rule", ["cc", "f2", "mw", "mwss", "gl", "dh"])
@pytest.mark.parametrize("n_points", [32, 64, 128])
@pytest.mark.parametrize("quadrature_module", [quadrature, quadrature_jax])
def test_quadrature_non_polynomial(f_and_integral, rule, n_points, quadrature_module):
    check_quadrature_rule(f_and_integral, rule, n_points, quadrature_module)


@pytest.mark.parametrize("sampling", ["cc", "f2", "mw", "mwss", "gl", "dh", "healpix"])
@pytest.mark.parametrize("L_or_nside", [1, 2, 3, 8, 10])
@pytest.mark.parametrize("quadrature_module", [quadrature, quadrature_jax])
def test_quadrature_weights(sampling, L_or_nside, quadrature_module):
    L, nside = (None, L_or_nside) if sampling == "healpix" else (L_or_nside, None)
    weights = quadrature_module.quad_weights(L=L, sampling=sampling, nside=nside)
    assert weights.shape[0] == samples.ntheta(L=L, sampling=sampling, nside=nside)
    # In general quadrature rules may use negative weights but
    # for all currently implemented schemes weights should be non-negative
    assert np.all(weights > 0)
    # Weights are equal for each longitude phi and so only computed for each theta
    # and rescaled to account for integration in phi - we expect the total sum of
    # weights (accounting for repeating by number of phi points) to be equal to
    # surface area of unit sphere - that is 4 * pi
    if sampling == "healpix":
        # HEALPix sampling scheme uses equiarea pixels with uniform weights
        assert np.all(weights == weights[0])
        assert np.isclose(
            weights[0] * samples.f_shape(L, sampling, nside)[0], 4 * np.pi
        )
    else:
        assert np.isclose(
            (weights * samples.nphi_equiang(L, sampling)).sum(), 4 * np.pi
        )
