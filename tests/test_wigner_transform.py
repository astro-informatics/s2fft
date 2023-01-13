import pytest
import so3
import numpy as np
import s2fft as s2f


L_to_test = [8, 16]
N_to_test = [2, 4, 6]
sampling_schemes_so3 = ["mw", "mwss"]
sampling_schemes = ["mw", "mwss", "dh"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
def test_inverse_wigner_transform(
    flmn_generator, s2fft_to_so3_sampling, L: int, N: int, sampling: str
):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )

    flmn_3D = flmn_generator(L=L, N=N, reality=False)
    flmn_1D = s2f.wigner.samples.flmn_3d_to_1d(flmn_3D, L, N)

    f_check = so3.inverse(flmn_1D, params)

    f = s2f.wigner.transform.inverse(flmn_3D, L, N, sampling)
    np.testing.assert_allclose(f.flatten("C"), f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes_so3)
def test_forward_wigner_transform(
    flmn_generator, s2fft_to_so3_sampling, L: int, N: int, sampling: str
):
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str=s2fft_to_so3_sampling(sampling)
    )

    flmn_3D = flmn_generator(L=L, N=N, reality=False)
    flmn_1D = s2f.wigner.samples.flmn_3d_to_1d(flmn_3D, L, N)

    f_1D = so3.inverse(flmn_1D, params)
    f_3D = f_1D.reshape(
        s2f.wigner.samples._ngamma(N),
        s2f.wigner.samples._nbeta(L, sampling),
        s2f.wigner.samples._nalpha(L, sampling),
    )

    flmn_check = s2f.wigner.samples.flmn_1d_to_3d(
        so3.forward(f_1D, params), L, N
    )
    flmn = s2f.wigner.transform.forward(f_3D, L, N, sampling)

    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_round_trip_wigner_transform(
    flmn_generator, L: int, N: int, sampling: str
):

    flmn = flmn_generator(L=L, N=N, reality=False)
    f = s2f.wigner.transform.inverse(flmn, L, N, sampling)
    flmn_check = s2f.wigner.transform.forward(f, L, N, sampling)

    np.testing.assert_allclose(flmn, flmn_check, atol=1e-14)
