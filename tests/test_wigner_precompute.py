import numpy as np
import pytest
import torch
from s2fft.precompute_transforms.wigner import inverse, forward
from s2fft.precompute_transforms import construct as c
from s2fft.base_transforms import wigner as base
from s2fft.sampling import so3_samples as samples
import so3

L_to_test = [6]
N_to_test = [2, 6]
nside_to_test = [4]
L_to_nside_ratio = [2]
reality_to_test = [False, True]
sampling_schemes = ["mw", "mwss", "dh", "gl"]
methods_to_test = ["numpy", "jax", "torch"]
modes_to_test = ["auto", "fft", "direct"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("mode", modes_to_test)
def test_inverse_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    mode: str,
):
    if mode.lower() == "fft" and sampling.lower() not in ["mw", "mwss", "dh"]:
        pytest.skip(
            f"Fourier based Wigner computation not valid for sampling={sampling}"
        )

    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality)

    kfunc = c.wigner_kernel_jax if method == "jax" else c.wigner_kernel
    kernel = kfunc(L, N, reality, sampling, forward=False, mode=mode)

    if method.lower() == "torch":
        # Test Transform
        f_check = inverse(
            torch.from_numpy(flmn),
            L,
            N,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
        )
        np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)

        # Test Gradients
        flmn_grad_test = torch.from_numpy(flmn)
        flmn_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            inverse,
            (
                flmn_grad_test,
                L,
                N,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
            ),
        )

    else:
        f_check = inverse(flmn, L, N, kernel, sampling, reality, method)
        np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("mode", modes_to_test)
def test_forward_wigner_transform(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    method: str,
    mode: str,
):
    if mode.lower() == "fft" and sampling.lower() not in ["mw", "mwss", "dh"]:
        pytest.skip(
            f"Fourier based Wigner computation not valid for sampling={sampling}"
        )
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, sampling=sampling, reality=reality)
    flmn = base.forward(f, L, N, sampling=sampling, reality=reality)

    kfunc = c.wigner_kernel_jax if method == "jax" else c.wigner_kernel
    kernel = kfunc(L, N, reality, sampling, forward=True, mode=mode)

    if method.lower() == "torch":
        # Test Transform
        flmn_check = forward(
            torch.from_numpy(f),
            L,
            N,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
        )
        np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)

        # Test Gradients
        f_grad_test = torch.from_numpy(f)
        f_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            forward,
            (
                f_grad_test,
                L,
                N,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
            ),
        )
    else:
        flmn_check = forward(f, L, N, kernel, sampling, reality, method)
        np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_wigner_transform_healpix(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality, nside)

    kfunc = c.wigner_kernel_jax if method == "jax" else c.wigner_kernel
    kernel = kfunc(L, N, reality, sampling, nside, forward=False)

    if method.lower() == "torch":
        # Test Transform
        f_check = inverse(
            torch.from_numpy(flmn),
            L,
            N,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
            nside,
        )
        np.testing.assert_allclose(
            np.real(f), np.real(f_check), atol=1e-5, rtol=1e-5
        )

        # Test Gradients
        flmn_grad_test = torch.from_numpy(flmn)
        flmn_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            inverse,
            (
                flmn_grad_test,
                L,
                N,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
                nside,
            ),
        )

    else:
        f_check = inverse(flmn, L, N, kernel, sampling, reality, method, nside)
        np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_wigner_transform_healpix(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
    method: str,
):
    sampling = "healpix"
    L = ratio * nside
    flmn = flmn_generator(L=L, N=N, reality=reality)

    f = base.inverse(flmn, L, N, 0, sampling, reality, nside)
    flmn_check = base.forward(f, L, N, 0, sampling, reality, nside)

    kfunc = c.wigner_kernel_jax if method == "jax" else c.wigner_kernel
    kernel = kfunc(L, N, reality, sampling, nside, forward=True)

    if method.lower() == "torch":
        # Test Transform
        flmn = forward(
            torch.from_numpy(f),
            L,
            N,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
            nside,
        )
        np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)

        # Test Gradients
        f_grad_test = torch.from_numpy(f)
        f_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            forward,
            (
                f_grad_test,
                L,
                N,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
                nside,
            ),
        )

    else:
        flmn = forward(f, L, N, kernel, sampling, reality, method, nside)
        np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("L", [8, 16, 32])
@pytest.mark.parametrize("fft_method", [True, False])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_wigner_transform_high_N(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    fft_method: bool,
    sampling: str,
    reality: bool,
):
    if sampling.lower() in ["gl", "dh"]:
        pytest.skip("SO3 benchmark only supports [mw, mwss] sampling.")

    N = int(L / np.log(L)) if fft_method else L

    flmn = flmn_generator(L=L, N=N, reality=reality)

    params = so3.create_parameter_dict(
        L=L,
        N=N,
        sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        reality=False,
    )
    f = so3.inverse(samples.flmn_3d_to_1d(flmn, L, N), params)

    kernel = c.wigner_kernel_jax(L, N, reality, sampling, forward=False)
    f_check = inverse(flmn, L, N, kernel, sampling, reality, "numpy")

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("L", [8, 16, 32])
@pytest.mark.parametrize("fft_method", [True, False])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_wigner_transform_high_N(
    flmn_generator,
    s2fft_to_so3_sampling,
    L: int,
    fft_method: bool,
    sampling: str,
    reality: bool,
):
    if sampling.lower() in ["gl", "dh"]:
        pytest.skip("SO3 benchmark only supports [mw, mwss] sampling.")

    N = int(L / np.log(L)) if fft_method else L

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

    kernel = c.wigner_kernel(L, N, reality, sampling, forward=True)
    flmn_check = forward(f_3D, L, N, kernel, sampling, reality, "numpy")

    np.testing.assert_allclose(flmn_so3, flmn_check, atol=1e-10, rtol=1e-10)
