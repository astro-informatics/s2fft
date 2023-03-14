import pytest
import so3
import numpy as np
from s2fft.sampling import so3_samples as samples

L_to_test = [8, 16]
N_to_test = [2, 4]
sampling_schemes = ["mw", "mwss"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_f_shape(s2fft_to_so3_sampling, L: int, N: int, sampling: str):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )
    fs = samples.f_shape(L, N, sampling)
    f_size = fs[0] * fs[1] * fs[2]
    assert f_size == so3.f_size(params)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_flmn_shape(s2fft_to_so3_sampling, L: int, N: int, sampling: str):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )
    assert samples.flmn_shape_1d(L, N) == so3.flmn_size(params)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_so3_samples(s2fft_to_so3_sampling, L: int, N: int, sampling: str):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )
    assert samples._nalpha(L, sampling) == so3.n_alpha(params)
    assert samples._nbeta(L, sampling) == so3.n_beta(params)
    assert samples._nalpha(N) == so3.n_gamma(params)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_elmn2ind(s2fft_to_so3_sampling, L: int, N: int, sampling: str):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )
    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                assert samples.elmn2ind(el, m, n, L, N) == so3.elmn2ind(
                    el, m, n, params
                )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_so3_samples(flmn_generator, L: int, N: int):
    flmn_3D = flmn_generator(L=L, N=N, reality=False)
    flmn_1D = samples.flmn_3d_to_1d(flmn_3D, L, N)
    flmn_3D_test = samples.flmn_1d_to_3d(flmn_1D, L, N)

    np.testing.assert_allclose(flmn_3D_test, flmn_3D, atol=1e-14)
