"""Tests for Clenshaw-Curtis and Fejer first-rule spherical transforms."""

import os
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from s2fft.recursions.price_mcewen import (
    generate_precomputes,
    generate_precomputes_jax,
)
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical
from s2fft.utils import quadrature, quadrature_jax, resampling, resampling_jax

jax.config.update("jax_enable_x64", True)


def _random_triangular_flm(L, spin, seed=1):
    rng = np.random.default_rng(seed)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for ell in range(abs(spin), L):
        flm[ell, L - 1 - ell : L + ell] = rng.normal(
            size=2 * ell + 1
        ) + 1j * rng.normal(size=2 * ell + 1)
    return flm


def _transform(value, L, spin, sampling, inverse, method="numpy", **kwargs):
    transform = spherical.inverse if inverse else spherical.forward
    return transform(
        value,
        L,
        spin=spin,
        sampling=sampling,
        method=method,
        **kwargs,
    )


def _roundtrip(flm, L, spin, sampling, method="numpy", reality=False, **kwargs):
    field = _transform(flm, L, spin, sampling, True, method, reality=reality, **kwargs)
    return field, _transform(field, L, spin, sampling, False, method, reality=reality)


def _random_complex(shape, rng):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _resampled_map(field, spin, sampling):
    ftm = np.fft.fftshift(np.fft.fft(field, axis=-1), axes=-1)
    ftm = resampling.resample_cc_f1_for_analysis(ftm, spin, sampling)
    return np.fft.ifft(np.fft.ifftshift(ftm, axes=-1), axis=-1)


def _real_triangular_flm(L, seed=1):
    flm = _random_triangular_flm(L, 0, seed)
    flm[:, L - 1] = flm[:, L - 1].real
    for m in range(1, L):
        flm[m:, L - 1 - m] = (-1) ** m * np.conj(flm[m:, L - 1 + m])
    return flm


def _numpy_precomputes(L, spin, betas, forward=False):
    precomps = generate_precomputes_jax(
        L, spin, "cc", forward=forward, betas=jnp.asarray(betas)
    )
    return [np.asarray(value) for value in precomps]


def _jacobi_values(degree, alpha, beta, x):
    """Evaluate a Jacobi polynomial with its standard three-term recurrence."""
    x = np.asarray(x)
    p0 = np.ones_like(x, dtype=np.float64)
    if degree == 0:
        return p0
    p1 = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)
    for n in range(2, degree + 1):
        two_n = 2 * n + alpha + beta
        denominator = 2 * n * (n + alpha + beta) * (two_n - 2)
        first = (two_n - 1) * (two_n * (two_n - 2) * x + alpha**2 - beta**2) * p1
        second = 2 * (n + alpha - 1) * (n + beta - 1) * two_n * p0
        p0, p1 = p1, (first - second) / denominator
    return p1


def _wigner_d_diagonal(ell, m, theta):
    """Evaluate d^ell_{m,m} independently using a Jacobi polynomial."""
    order = abs(m)
    return np.cos(theta / 2) ** (2 * order) * _jacobi_values(
        ell - order, 0, 2 * order, np.cos(theta)
    )


def _diagonal_mode_map(ell, m, value, L, spin, sampling, ntheta=None):
    """Evaluate an m=-spin mode from its Jacobi-polynomial Wigner d."""
    if ntheta is None:
        theta = samples.thetas(L, sampling)
    elif sampling == "cc":
        theta = np.linspace(0, np.pi, ntheta)
    else:
        theta = (np.arange(ntheta) + 0.5) * np.pi / ntheta
    phi = samples.phis_equiang(L, sampling)
    return (
        (-1) ** spin
        * np.sqrt((2 * ell + 1) / (4 * np.pi))
        * value
        * _wigner_d_diagonal(ell, m, theta)[:, None]
        * np.exp(1j * m * phi[None, :])
    )


def _pack_ducc_real_alm(flm, L):
    """Pack S2FFT real-field coefficients in DUCC's positive-m healpy order."""
    alm = np.zeros((1, L * (L + 1) // 2), dtype=np.complex128)
    for m in range(L):
        for ell in range(m, L):
            index = m * (2 * L - 1 - m) // 2 + ell
            alm[0, index] = flm[ell, L - 1 + m]
    return alm


def _unpack_ducc_real_alm(alm, L):
    """Restore S2FFT centered-m order and real-field negative-m symmetry."""
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for m in range(L):
        for ell in range(m, L):
            index = m * (2 * L - 1 - m) // 2 + ell
            value = alm[0, index]
            flm[ell, L - 1 + m] = value
            if m:
                flm[ell, L - 1 - m] = (-1) ** m * np.conj(value)
    return flm


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("L", [4, 8])
def test_default_geometry(sampling, L):
    ntheta = 2 * L - 1 if sampling == "cc" else L
    assert samples.ntheta(L, sampling) == ntheta
    assert samples.nphi_equiang(L, sampling) == 2 * L
    assert samples.ftm_shape(L, sampling) == (ntheta, 2 * L)
    assert samples.f_shape(L, sampling) == (ntheta, 2 * L)
    expected = (
        np.linspace(0, np.pi, ntheta)
        if sampling == "cc"
        else (np.arange(ntheta) + 0.5) * np.pi / ntheta
    )
    np.testing.assert_allclose(samples.thetas(L, sampling), expected)
    np.testing.assert_allclose(
        samples.phis_equiang(L, sampling), np.arange(2 * L) * np.pi / L
    )


@pytest.mark.parametrize("L", [4, 8])
@pytest.mark.parametrize("spin", [-1, 0, 1])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_f1_forward_precomputes(L, spin, method):
    generate = generate_precomputes if method == "numpy" else generate_precomputes_jax

    forward_precomps = generate(L, spin, sampling="f1", forward=True)
    assert np.asarray(forward_precomps[0]).shape[1] == 2 * L - 1
    assert np.asarray(forward_precomps[4]).shape[0] == 2 * L - 1

    field = _random_complex(samples.f_shape(L, "f1"), np.random.default_rng(L + spin))
    expected = spherical.forward(field, L, spin=spin, sampling="f1", method=method)
    actual = spherical.forward(
        field,
        L,
        spin=spin,
        sampling="f1",
        method=method,
        precomps=forward_precomps,
    )
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)

    inverse_precomps = generate(L, spin, sampling="f1", forward=False)
    assert np.asarray(inverse_precomps[0]).shape[1] == L

    explicit_betas = samples.thetas(L, "f1")
    explicit_forward = generate(
        L, spin, sampling="f1", forward=True, betas=explicit_betas
    )
    assert np.asarray(explicit_forward[0]).shape[1] == len(explicit_betas)

    cc_forward = generate(L, spin, sampling="cc", forward=True)
    assert np.asarray(cc_forward[0]).shape[1] == 2 * L - 3
    cc_field = _random_complex(
        samples.f_shape(L, "cc"), np.random.default_rng(100 + L + spin)
    )
    cc_expected = spherical.forward(
        cc_field, L, spin=spin, sampling="cc", method=method
    )
    cc_actual = spherical.forward(
        cc_field,
        L,
        spin=spin,
        sampling="cc",
        method=method,
        precomps=cc_forward,
    )
    np.testing.assert_allclose(cc_actual, cc_expected, atol=3e-11, rtol=3e-11)

    cc_inverse = generate(L, spin, sampling="cc", forward=False)
    assert np.asarray(cc_inverse[0]).shape[1] == 2 * L - 1


# High-bandlimit quadrature is inexpensive; full DUCC transform comparisons
# at L >= 512 remain opt-in because of their much greater runtime.
@pytest.mark.parametrize("L", [1, 4, 8, 128, 512, 1024])
@pytest.mark.parametrize("module", [quadrature, quadrature_jax])
def test_f1_quadrature(L, module):
    w = np.asarray(module.quad_weights_f1_theta_only(L))
    x = np.cos((np.arange(L) + 0.5) * np.pi / L)
    for degree in range(L):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        np.testing.assert_allclose(np.dot(w, x**degree), expected, atol=2e-13)
    np.testing.assert_allclose(
        np.sum(np.asarray(module.quad_weights_f1(L))) * 2 * L, 4 * np.pi
    )


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("n_phi", [7, 8])
@pytest.mark.parametrize("spin", [-1, 1, 2])
def test_meridian_interpolation(sampling, n_phi, spin):
    n_theta = 6
    ftm = _random_complex((n_theta, n_phi), np.random.default_rng(8))
    dense = resampling.resample_cc_f1_for_analysis(ftm, spin, sampling)
    dense_jax = resampling_jax.resample_cc_f1_for_analysis(
        jnp.asarray(ftm), spin, sampling
    )
    np.testing.assert_allclose(dense_jax, dense, atol=2e-13)
    if sampling == "cc":
        np.testing.assert_allclose(dense[::2], ftm, atol=2e-13)
    else:
        np.testing.assert_allclose(dense[1::2], ftm, atol=2e-13)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("spin", [-1, 1])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_even_longitude_analytic_spin_mode(sampling, spin, method):
    L, ell, m, value = 4, 3, -spin, 0.37 - 0.22j
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    flm[ell, L - 1 + m] = value
    field, recovered = _roundtrip(flm, L, spin, sampling, method)
    assert field.shape == samples.f_shape(L, sampling)
    np.testing.assert_allclose(
        field,
        _diagonal_mode_map(ell, m, value, L, spin, sampling),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(recovered, flm, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("L", [4, 8])
@pytest.mark.parametrize("spin", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_default_grid_roundtrip(sampling, L, spin, method):
    flm = _random_triangular_flm(L, spin, 10 + L + spin)
    field, recovered = _roundtrip(flm, L, spin, sampling, method)
    assert field.shape == samples.f_shape(L, sampling)
    np.testing.assert_allclose(recovered, flm, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("spin", [-1, 0, 1])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reduced_cc_roundtrip(spin, method):
    L = 8
    flm = _random_triangular_flm(L, spin, 80 + spin)
    full_field, full_recovered = _roundtrip(flm, L, spin, "cc", method)
    reduced_field = spherical.inverse(
        flm, L, spin=spin, sampling="cc", method=method, ntheta=L + 1
    )
    reduced_recovered = spherical.forward(
        reduced_field, L, spin=spin, sampling="cc", method=method
    )

    assert full_field.shape == (2 * L - 1, 2 * L)
    assert reduced_field.shape == (L + 1, 2 * L)
    np.testing.assert_allclose(
        reduced_recovered, full_recovered, atol=3e-11, rtol=3e-11
    )
    np.testing.assert_allclose(reduced_recovered, flm, atol=3e-11, rtol=3e-11)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reduced_cc_forward_precomputes_use_dense_cc_nodes():
    L, spin = 8, 1
    flm = _random_triangular_flm(L, spin, 92)
    field = spherical.inverse(flm, L, spin=spin, sampling="cc", ntheta=L + 1)
    betas = samples.thetas(L + 1, "cc")[1:-1]
    precomps = _numpy_precomputes(L, spin, betas, forward=True)
    recovered = spherical.forward(field, L, spin=spin, sampling="cc", precomps=precomps)
    np.testing.assert_allclose(recovered, flm, atol=3e-11, rtol=3e-11)

    default_precomps = _numpy_precomputes(
        L, spin, samples.thetas(L, "cc")[1:-1], forward=True
    )
    with pytest.raises(ValueError, match="analysis CC latitude nodes"):
        spherical.forward(field, L, spin=spin, sampling="cc", precomps=default_precomps)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_high_bandlimit_nonzero_spin_modes(sampling):
    L, spin = 72, 1
    ntheta = L + 1 if sampling == "cc" else L
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    field = np.zeros((ntheta, 2 * L), dtype=np.complex128)
    for ell, value in ((L - 2, 0.3 - 0.2j), (L - 1, -0.1 + 0.4j)):
        flm[ell, L - 1 - spin] = value
        field += _diagonal_mode_map(ell, -spin, value, L, spin, sampling, ntheta=ntheta)

    recovered = spherical.forward(field, L, spin=spin, sampling=sampling)
    np.testing.assert_allclose(recovered, flm, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reduced_high_bandlimit_matches_ducc(sampling):
    ducc_sht = pytest.importorskip("ducc0.sht")
    L = 72
    ntheta = L + 1 if sampling == "cc" else L
    nphi = 2 * L
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for ell, m, value in (
        (0, 0, 1.0),
        (L - 1, 0, 0.31),
        (L - 2, 1, 0.17 + 0.08j),
        (L - 1, L - 1, -0.12 + 0.05j),
    ):
        flm[ell, L - 1 + m] = value
        if m:
            flm[ell, L - 1 - m] = (-1) ** m * np.conj(value)
    alm = _pack_ducc_real_alm(flm, L)
    field = ducc_sht.synthesis_2d(
        alm=alm,
        spin=0,
        lmax=L - 1,
        geometry=sampling.upper(),
        ntheta=ntheta,
        nphi=nphi,
    )[0]
    recovered = spherical.forward(field, L, sampling=sampling, reality=True)
    np.testing.assert_allclose(recovered, flm, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("sampling,ntheta", [("cc", 8), ("f1", 7)])
def test_cc_f1_analysis_rejects_undersampled_geometry(sampling, ntheta):
    L = 8
    field = np.zeros((ntheta, 2 * L), dtype=np.complex128)
    with pytest.raises(ValueError, match="analysis requires ntheta"):
        spherical.forward(field, L, sampling=sampling)

    ntheta = L + 1 if sampling == "cc" else L
    field = np.zeros((ntheta, 2 * L - 1))
    with pytest.raises(ValueError, match="standard nphi=2L"):
        spherical.forward(field, L, sampling=sampling)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reduced_cc_jax_jit_and_differentiation():
    L, spin = 4, 1
    rng = np.random.default_rng(95)
    shape = (L + 1, 2 * L)
    value = jnp.asarray(_random_complex(shape, rng))
    tangent = jnp.asarray(_random_complex(shape, rng))
    target = jnp.asarray(_random_complex((L, 2 * L - 1), rng))

    def transform(x):
        return spherical.forward_jax(x, L, spin=spin, sampling="cc")

    _check_jax_linear_transform(
        transform,
        value,
        tangent,
        target,
        lambda x: spherical.forward_numpy(x, L, spin=spin, sampling="cc"),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reduced_cc_iterative_refinement_keeps_input_geometry():
    L, spin = 8, -1
    flm = _random_triangular_flm(L, spin, 96)
    field = spherical.inverse(flm, L, spin=spin, sampling="cc", ntheta=L + 1)
    recovered = spherical.forward(field, L, spin=spin, sampling="cc", iter=1)
    assert field.shape == (L + 1, 2 * L)
    np.testing.assert_allclose(recovered, flm, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.parametrize("reality", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_refinement(sampling, method, reality):
    L = 4
    flm = _real_triangular_flm(L) if reality else _random_triangular_flm(L, 0)
    field = _transform(flm, L, 0, sampling, True, method, reality=reality)
    for steps in (0, 1, 2):
        recovered = _transform(
            field, L, 0, sampling, False, method, reality=reality, iter=steps
        )
        np.testing.assert_allclose(recovered, flm, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.parametrize("reality", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_single_precision(sampling, method, reality):
    L = 4
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex64)
    flm[2, L - 1 if reality else L] = 0.5 if reality else 0.3 + 0.1j
    context = jax.experimental.enable_x64(False) if method == "jax" else nullcontext()
    with context:
        field, recovered = _roundtrip(
            jnp.asarray(flm) if method == "jax" else flm,
            L,
            0,
            sampling,
            method,
            reality,
        )
    assert field.shape == samples.f_shape(L, sampling)
    np.testing.assert_allclose(recovered, flm, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_weighted_adjoint(sampling):
    L, spin = 8, 1
    flm = _random_triangular_flm(L, spin, 32)
    field = _random_complex(samples.f_shape(L, sampling), np.random.default_rng(31))
    synthesis = _transform(flm, L, spin, sampling, True)
    coefficients = _transform(field, L, spin, sampling, False)
    if sampling == "f1":
        field, synthesis = (
            _resampled_map(value, spin, sampling) for value in (field, synthesis)
        )
        weights = quadrature.quad_weights_cc_theta_only(L + 1) * (np.pi / L)
    else:
        weights = quadrature.quad_weights_cc(L)
    np.testing.assert_allclose(
        np.sum(np.conj(synthesis) * field * weights[:, None]),
        np.vdot(flm, coefficients),
        atol=3e-11,
        rtol=3e-11,
    )


def _check_jax_linear_transform(transform, value, tangent, target, reference):
    output = jax.jit(transform)(value)
    np.testing.assert_allclose(output, reference(value), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        jax.vmap(transform)(jnp.stack([value, value])),
        jnp.stack([transform(value), transform(value)]),
        atol=2e-12,
    )
    output, output_tangent = jax.jvp(transform, (value,), (tangent,))
    np.testing.assert_allclose(output_tangent, transform(tangent), atol=2e-12)
    _, pullback = jax.vjp(transform, value)
    transpose = pullback(target)[0]
    # JAX transposes complex linear maps with a bilinear convention.
    np.testing.assert_allclose(
        jnp.sum(target * output), jnp.sum(transpose * value), atol=2e-11
    )

    def loss(x):
        return jnp.real(jnp.vdot(transform(x), target))

    gradient = jax.grad(loss)(value)
    step = 1e-6
    finite_difference = (
        loss(value + step * tangent) - loss(value - step * tangent)
    ) / (2 * step)
    np.testing.assert_allclose(
        jnp.real(jnp.sum(gradient * tangent)), finite_difference, rtol=2e-8, atol=2e-8
    )


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("spin", [-1, 1])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_jax_differentiable(sampling, spin, direction):
    L = 4
    rng = np.random.default_rng(12 + spin)
    if direction == "inverse":
        value = jnp.asarray(_random_triangular_flm(L, spin, 21))
        tangent = jnp.asarray(_random_triangular_flm(L, spin, 22))
        target_shape = samples.f_shape(L, sampling)
    else:
        value = jnp.asarray(_random_complex(samples.f_shape(L, sampling), rng))
        tangent = jnp.asarray(_random_complex(samples.f_shape(L, sampling), rng))
        target_shape = (L, 2 * L - 1)
    target = jnp.asarray(_random_complex(target_shape, rng))
    transform = getattr(spherical, f"{direction}_jax")
    reference = getattr(spherical, f"{direction}_numpy")

    def apply(x):
        return transform(x, L, spin=spin, sampling=sampling)

    def ref(x):
        return reference(np.asarray(x), L, spin=spin, sampling=sampling)

    _check_jax_linear_transform(apply, value, tangent, target, ref)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("L", [72, 128])
@pytest.mark.parametrize("spin", [-2, -1, 1, 2])
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_high_bandlimit_spin(sampling, L, spin):
    flm = _random_triangular_flm(L, spin, 100 + L + spin)
    _, recovered = _roundtrip(flm, L, spin, sampling)
    np.testing.assert_allclose(recovered, flm, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("sampling", ["cc", "f1"])
@pytest.mark.parametrize("L", [4, 72, 128, 512, 1024])
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_modes_match_ducc(sampling, L):
    if L >= 512 and os.environ.get("S2FFT_RUN_VERY_HIGH_L") != "1":
        pytest.skip("Set S2FFT_RUN_VERY_HIGH_L=1 to run L >= 512 cases")
    ducc_sht = pytest.importorskip("ducc0.sht")
    ntheta, nphi = samples.f_shape(L, sampling)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for ell, m in ((0, 0), (L - 1, 0), (L - 2, 1), (L - 1, L - 2), (L - 1, L - 1)):
        value = 1.0 if (ell, m) == (0, 0) else 0.31 * (ell + 1) / L
        if m:
            value += 0.17j * (ell + 1) / L
        flm[ell, L - 1 + m] = value
        if m:
            flm[ell, L - 1 - m] = (-1) ** m * np.conj(value)
    alm = _pack_ducc_real_alm(flm, L)
    ducc_map = ducc_sht.synthesis_2d(
        alm=alm,
        spin=0,
        lmax=L - 1,
        geometry=sampling.upper(),
        ntheta=ntheta,
        nphi=nphi,
    )[0]
    inverse_precomps = _numpy_precomputes(L, 0, samples.thetas(L, sampling))
    forward_betas = (
        samples.thetas(L + 1, "cc")[1:-1]
        if sampling == "f1"
        else samples.thetas(L, "cc")[1:-1]
    )
    forward_precomps = _numpy_precomputes(L, 0, forward_betas, forward=True)
    dims = dict(L=L, spin=0, sampling=sampling, reality=True)
    synthesized = spherical.inverse_numpy(flm, **dims, precomps=inverse_precomps)
    recovered = spherical.forward_numpy(ducc_map, **dims, precomps=forward_precomps)
    ducc_recovered = ducc_sht.analysis_2d(
        map=ducc_map[None, ...],
        spin=0,
        lmax=L - 1,
        mmax=L - 1,
        geometry=sampling.upper(),
    )
    tolerance = 5e-9 if L >= 1024 else 2e-10 if L >= 128 else 2e-11
    np.testing.assert_allclose(synthesized, ducc_map, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(recovered, flm, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        _unpack_ducc_real_alm(ducc_recovered, L),
        flm,
        atol=tolerance,
        rtol=tolerance,
    )
