import jax
import numpy as np
import pytest
import so3

from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms import fourier_wigner as fw
from s2fft.sampling import so3_samples as samples

jax.config.update("jax_enable_x64", True)

# Test cases
L_to_test = [16]
N_to_test = [2, 8, 16]
reality_to_test = [False, True]
sampling_schemes = ["mw", "mwss"]
methods_to_test = ["numpy", "jax"]
delta_method_to_test = ["otf", "precomp"]

# Test tolerance
atol = 1e-12


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("delta_method", delta_method_to_test)
def test_inverse_fourier_wigner_transform(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    delta_method: str,
):
    flmn = flmn_generator(L=L, N=N, reality=reality)

    params = so3.create_parameter_dict(
        L=L,
        N=N,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )
    f = so3.inverse(samples.flmn_3d_to_1d(flmn, L, N), params)

    delta = None
    transform = fw.inverse_transform_jax if method == "jax" else fw.inverse_transform
    if delta_method.lower() == "precomp":
        delta = (
            c.fourier_wigner_kernel_jax(L)
            if method == "jax"
            else c.fourier_wigner_kernel(L)
        )
    f_check = transform(flmn, L, N, delta, reality, sampling)
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=atol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("delta_method", delta_method_to_test)
def test_forward_fourier_wigner_transform(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    delta_method: str,
):
    flmn = flmn_generator(L=L, N=N, reality=reality)

    params = so3.create_parameter_dict(
        L=L,
        N=N,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )
    f = so3.inverse(samples.flmn_3d_to_1d(flmn, L, N), params)
    f_3D = f.reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )
    flmn = samples.flmn_1d_to_3d(so3.forward(f, params), L, N)

    delta = None
    transform = fw.forward_transform_jax if method == "jax" else fw.forward_transform
    if delta_method.lower() == "precomp":
        delta = (
            c.fourier_wigner_kernel_jax(L)
            if method == "jax"
            else c.fourier_wigner_kernel(L)
        )
    flmn_check = transform(f_3D, L, N, delta, reality, sampling)
    np.testing.assert_allclose(flmn, flmn_check, atol=atol)


@pytest.mark.parametrize("L", [8, 16, 32, 64])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_fourier_wigner_transform_high_N(
    flmn_generator, s2fft_to_so3_sampling, L: int, sampling: str, reality: bool
):
    N = L
    flmn = flmn_generator(L=L, N=N, reality=reality)

    params = so3.create_parameter_dict(
        L=L,
        N=N,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )
    f = so3.inverse(samples.flmn_3d_to_1d(flmn, L, N), params)

    f = f.real if reality else f
    delta = c.fourier_wigner_kernel(L)
    f_check = fw.inverse_transform(flmn, L, N, delta, reality, sampling)

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=atol)


@pytest.mark.parametrize("L", [8, 16, 32, 64])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_fourier_wigner_transform_high_N(
    flmn_generator, s2fft_to_so3_sampling, L: int, sampling: str, reality: bool
):
    N = L
    flmn = flmn_generator(L=L, N=N, reality=reality)

    params = so3.create_parameter_dict(
        L=L,
        N=N,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )

    f_1D = so3.inverse(samples.flmn_3d_to_1d(flmn, L, N), params)
    f_3D = f_1D.reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )
    flmn_so3 = samples.flmn_1d_to_3d(so3.forward(f_1D, params), L, N)

    delta = c.fourier_wigner_kernel_jax(L)
    flmn_check = fw.forward_transform_jax(f_3D, L, N, delta, reality, sampling)
    np.testing.assert_allclose(flmn_so3, flmn_check, atol=atol)
