import pytest
import so3
import numpy as np
from s2fft.sampling import so3_samples as samples
from s2fft.base_transforms import wigner


L_to_test = [6, 7]
N_to_test = [2, 3]
L_lower_to_test = [0, 2]
sampling_schemes_so3 = ["mw", "mwss"]
sampling_schemes = ["mw", "mwss", "dh"]
reality_to_test = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_wigner_transform(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    reality: bool,
):
    params = so3.create_parameter_dict(
        L=L,
        N=N,
        L0=L_lower,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )

    flmn_3D = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    flmn_1D = samples.flmn_3d_to_1d(flmn_3D, L, N)

    f_check = so3.inverse(flmn_1D, params)

    f = wigner.inverse(flmn_3D, L, N, L_lower, sampling, reality)
    f = f.flatten("C")

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_wigner_transform(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    N: int,
    L_lower: int,
    sampling: str,
    reality: bool,
):
    params = so3.create_parameter_dict(
        L=L,
        N=N,
        L0=L_lower,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
    )

    flmn_3D = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)
    flmn_1D = samples.flmn_3d_to_1d(flmn_3D, L, N)

    f_1D = so3.inverse(flmn_1D, params)
    f_3D = f_1D.reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )

    flmn_check = samples.flmn_1d_to_3d(so3.forward(f_1D, params), L, N)
    flmn = wigner.forward(f_3D, L, N, L_lower, sampling, reality)

    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)


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
