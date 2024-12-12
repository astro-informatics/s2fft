import numpy as np
import pytest

import s2fft
import s2fft.sampling as smp
import s2fft.utils.signal_generator as gen

L_values_to_test = [4, 7, 64]
L_lower_to_test = [0]
spin_to_test = [-2, 0, 1]
reality_values_to_test = [False, True]


@pytest.mark.parametrize("L", L_values_to_test)
@pytest.mark.parametrize("min_el", [0, 1])
def test_complex_el_and_m_indices(L, min_el):
    expected_el_indices, expected_m_indices = np.array(
        [(el, m) for el in range(min_el, L) for m in range(1, el + 1)]
    ).T
    el_indices, m_indices = gen.complex_el_and_m_indices(L, min_el)
    assert (el_indices == expected_el_indices).all()
    assert (m_indices == expected_m_indices).all()


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
    if reality:
        check_flm_conjugate_symmetry(flm, L, max(L_lower, abs(spin)))
        f_complex = s2fft.inverse(flm, L, spin=spin, reality=False, L_lower=L_lower)
        assert np.allclose(f_complex.imag, 0)
        f_real = s2fft.inverse(flm, L, spin=spin, reality=True, L_lower=L_lower)
        assert np.allclose(f_complex.real, f_real)
