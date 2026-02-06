from collections.abc import Callable

import numpy as np
import pytest

from s2fft.base_transforms import wigner
from s2fft.sampling import so3_samples as samples

L_to_test = [6, 7]
N_to_test = [2, 3]
L_lower_to_test = [0, 2]
sampling_schemes_so3 = ["mw", "mwss"]
sampling_schemes = ["mw", "mwss", "dh", "gl"]
reality_to_test = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_wigner_transform(
    cached_so3_test_case: Callable,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    reality: bool,
):
    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=L_lower, sampling=sampling, reality=reality
    )

    f = wigner.inverse(test_data["flmn"], L, N, L_lower, sampling, reality)
    f = f.flatten("C")

    np.testing.assert_allclose(f, test_data["f_so3"], atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_wigner_transform(
    cached_so3_test_case: Callable,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    reality: bool,
):
    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=L_lower, sampling=sampling, reality=reality
    )

    f = test_data["f_so3"].reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )
    flmn = wigner.forward(f, L, N, L_lower, sampling, reality)

    np.testing.assert_allclose(flmn, test_data["flmn_so3"], atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_round_trip_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    reality: bool,
):
    flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    f = wigner.inverse(flmn, L, N, L_lower, sampling, reality=reality)
    flmn_check = wigner.forward(f, L, N, L_lower, sampling, reality=reality)

    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)
