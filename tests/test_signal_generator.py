import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import s2fft
import s2fft.sampling as smp
import s2fft.utils.signal_generator as gen

L_values_to_test = [6, 7, 16]
L_lower_to_test = [0, 1]
spin_to_test = [-2, 0, 1]
reality_values_to_test = [False, True]


@pytest.mark.parametrize("size", (10, 100, 1000))
@pytest.mark.parametrize("var", (1, 2))
def test_complex_normal(rng, size, var):
    samples = gen.complex_normal(rng, size, var)
    assert samples.dtype == np.complex128
    assert samples.size == size
    mean = samples.mean()
    # Error in real + imag components of mean estimate ~ Normal(0, (var / 2) / size)
    # Therefore difference between mean estimate and true zero value should be
    # less than 3 * sqrt(var / (2 * size)) with probability 0.997
    mean_error_tol = 3 * (var / (2 * size)) ** 0.5
    assert abs(mean.imag) < mean_error_tol and abs(mean.real) < mean_error_tol
    # If S is (unbiased) sample variance estimate then (size - 1) * S / var is a
    # chi-squared distributed random variable with (size - 1) degrees of freedom
    # For size >> 1, S ~approx Normal(var, 2 * var**2 / (size - 1)) so error in
    # variance estimate should be less than 3 * sqrt(2 * var**2 / (size - 1))
    # with high probability
    assert abs(samples.var(ddof=1) - var) < 3 * (2 * var**2 / (size - 1)) ** 0.5


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("min_el", [0, 1])
def test_complex_el_and_m_indices(L, min_el):
    expected_el_indices, expected_m_indices = np.array(
        [(el, m) for el in range(min_el, L) for m in range(1, el + 1)]
    ).T
    el_indices, m_indices = gen.complex_el_and_m_indices(L, min_el)
    assert (el_indices == expected_el_indices).all()
    assert (m_indices == expected_m_indices).all()


def check_flm_zeros(flm, L, min_el):
    for el in range(L):
        for m in range(L):
            if el < min_el or m > el:
                assert flm[el, L - 1 + m] == flm[el, L - 1 - m] == 0


def check_flm_conjugate_symmetry(flm, L, min_el):
    for el in range(min_el, L):
        for m in range(el + 1):
            assert flm[el, L - 1 - m] == (-1) ** m * flm[el, L - 1 + m].conj()


def check_flm_unequal(flm1, flm2, L, min_el):
    """assert that two passed flm are elementwise unequal"""
    for el in range(L):
        for m in range(L):
            if not (el < min_el or m > el):
                assert flm1[el, L - 1 + m] != flm2[el, L - 1 - m]


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("reality", reality_values_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_generate_flm(rng, L, L_lower, spin, reality):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    flm = gen.generate_flm(rng, L, L_lower, spin, reality)
    assert flm.shape == smp.s2_samples.flm_shape(L)
    assert flm.dtype == np.complex128
    assert np.isfinite(flm).all()
    check_flm_zeros(flm, L, max(L_lower, abs(spin)))
    if reality:
        check_flm_conjugate_symmetry(flm, L, max(L_lower, abs(spin)))
        f_complex = s2fft.inverse(flm, L, spin=spin, reality=False, L_lower=L_lower)
        assert np.allclose(f_complex.imag, 0)
        f_real = s2fft.inverse(flm, L, spin=spin, reality=True, L_lower=L_lower)
        assert np.allclose(f_complex.real, f_real)


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("reality", reality_values_to_test)
def test_generate_flm_size(rng, L, L_lower, spin, reality):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    size = 2
    flm = gen.generate_flm(rng, L, L_lower, spin, reality, size=size)
    assert flm.shape == (size,) + smp.s2_samples.flm_shape(L)
    check_flm_zeros(flm[0], L, max(L_lower, abs(spin)))
    check_flm_zeros(flm[1], L, max(L_lower, abs(spin)))
    check_flm_unequal(flm[0], flm[1], L, max(L_lower, abs(spin)))

    size = (3, 4)
    flm = gen.generate_flm(rng, L, L_lower, spin, reality, size=size)
    assert flm.shape == size + smp.s2_samples.flm_shape(L)


def check_flmn_zeros(flmn, L, N, L_lower):
    for n in range(-N + 1, N):
        min_el = max(L_lower, abs(n))
        for el in range(L):
            for m in range(L):
                if el < min_el or m > el:
                    assert (
                        flmn[N - 1 + n, el, L - 1 + m]
                        == flmn[N - 1 + n, el, L - 1 - m]
                        == 0
                    )


def check_flmn_conjugate_symmetry(flmn, L, N, L_lower):
    for n in range(-N + 1, N):
        min_el = max(L_lower, abs(n))
        for el in range(min_el, L):
            for m in range(el + 1):
                assert (
                    flmn[N - 1 - n, el, L - 1 - m]
                    == (-1) ** (m + n) * flmn[N - 1 + n, el, L - 1 + m].conj()
                )


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("reality", reality_values_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_generate_flmn(rng, L, N, L_lower, reality):
    flmn = gen.generate_flmn(rng, L, N, L_lower, reality)
    assert flmn.shape == smp.so3_samples.flmn_shape(L, N)
    assert flmn.dtype == np.complex128
    assert np.isfinite(flmn).all()
    check_flmn_zeros(flmn, L, N, L_lower)
    if reality:
        check_flmn_conjugate_symmetry(flmn, L, N, L_lower)
        f_complex = s2fft.wigner.inverse(flmn, L, N, reality=False, L_lower=L_lower)
        assert np.allclose(f_complex.imag, 0)
        f_real = s2fft.wigner.inverse(flmn, L, N, reality=True, L_lower=L_lower)
        assert np.allclose(f_complex.real, f_real)


def gaussian_covariance(spectra):
    """Gaussian covariance for a stack of spectra.

    If the shape of *spectra* is *(K, K, L)*, the shape of the
    covariance is *(L, C, C)*, where ``C = K * (K + 1) // 2``
    is the number of independent spectra.

    """
    _, K, L = spectra.shape
    row, col = np.tril_indices(K)
    cov = np.zeros((L, row.size, col.size))
    ell = np.arange(L)
    for i, j in np.ndindex(row.size, col.size):
        cov[:, i, j] = (
            spectra[row[i], row[j]] * spectra[col[i], col[j]]
            + spectra[row[i], col[j]] * spectra[col[i], row[j]]
        ) / (2 * ell + 1)
    return cov


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("xp", [np, jnp])
def test_generate_flm_from_spectra(rng, L, xp):
    # number of fields to generate
    K = 4

    # correlation matrix for fields, applied to all ell
    corr = xp.asarray(
        [
            [1.0, 0.1, -0.1, 0.1],
            [0.1, 1.0, 0.1, -0.1],
            [-0.1, 0.1, 1.0, 0.1],
            [0.1, -0.1, 0.1, 1.0],
        ],
    )

    ell = xp.arange(L)

    # auto-spectra are power laws
    powers = xp.arange(1, K + 1)
    auto = 1 / (2 * ell + 1) ** powers[:, None]

    # compute the spectra from auto and corr
    spectra = xp.sqrt(auto[:, None, :] * auto[None, :, :]) * corr[:, :, None]
    assert spectra.shape == (K, K, L)

    # generate random flm from spectra
    flm = s2fft.utils.signal_generator.generate_flm_from_spectra(rng, spectra)
    assert flm.shape == (K, L, 2 * L - 1)

    # compute the realised spectra
    re, im = flm.real, flm.imag
    result = (
        re[None, :, :, :] * re[:, None, :, :] + im[None, :, :, :] * im[:, None, :, :]
    )
    result = result.sum(axis=-1) / (2 * ell + 1)

    # compute covariance of sampled spectra
    cov = gaussian_covariance(spectra)

    # data vector, remove duplicate entries, and put L dim first
    x = result - spectra
    x = x[np.tril_indices(K)]
    x = x.T

    # compute chi2/n of realised spectra
    y = xp.linalg.solve(cov, x[..., None])[..., 0]
    n = x.size
    chi2_n = (x * y).sum() / n

    # make sure chi2/n is as expected
    sigma = np.sqrt(2 / n)
    assert np.fabs(chi2_n - 1.0) < 3 * sigma


@pytest.mark.parametrize("L", L_values_to_test)
def test_generate_flm_from_spectra_grads(L):
    # fixed set of power spectra
    ell = jnp.arange(L)
    cl = 1 / (2 * ell + 1)
    spectra = cl.reshape(1, 1, L)

    def func(x):
        rng = np.random.default_rng(42)
        return s2fft.utils.signal_generator.generate_flm_from_spectra(rng, x)

    check_grads(func, (spectra,), 1)
