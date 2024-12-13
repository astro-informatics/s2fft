import numpy as np
import pytest

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
