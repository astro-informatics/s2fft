from collections.abc import Callable

import numpy as np
import pytest

from s2fft.sampling import so3_samples as samples

L_to_test = [8, 16]
N_to_test = [2, 4]
sampling_schemes = ["mw", "mwss"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_f_shape(cached_so3_samples_test_case: Callable, L: int, N: int, sampling: str):
    test_data = cached_so3_samples_test_case(L=L, N=N, sampling=sampling)
    fs = samples.f_shape(L, N, sampling)
    f_size = fs[0] * fs[1] * fs[2]
    assert f_size == test_data["f_size"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_flmn_shape(
    cached_so3_samples_test_case: Callable, L: int, N: int, sampling: str
):
    test_data = cached_so3_samples_test_case(L=L, N=N, sampling=sampling)
    assert samples.flmn_shape_1d(L, N) == test_data["flmn_size"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_so3_samples(
    cached_so3_samples_test_case: Callable, L: int, N: int, sampling: str
):
    test_data = cached_so3_samples_test_case(L=L, N=N, sampling=sampling)
    assert samples._nalpha(L, sampling) == test_data["n_alpha"]
    assert samples._nbeta(L, sampling) == test_data["n_beta"]
    assert samples._nalpha(N) == test_data["n_gamma"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_elmn2ind(
    cached_so3_samples_test_case: Callable, L: int, N: int, sampling: str
):
    test_data = cached_so3_samples_test_case(L=L, N=N, sampling=sampling)
    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                assert (
                    samples.elmn2ind(el, m, n, L, N)
                    == test_data["elmn2ind"][f"{el}_{m}_{n}"]
                )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_so3_samples_2(flmn_generator, L: int, N: int):
    flmn_3D = flmn_generator(L=L, N=N, reality=False)
    flmn_1D = samples.flmn_3d_to_1d(flmn_3D, L, N)
    flmn_3D_test = samples.flmn_1d_to_3d(flmn_1D, L, N)

    np.testing.assert_allclose(flmn_3D_test, flmn_3D, atol=1e-14)
