from collections.abc import Callable

import jax
import numpy as np
import pytest
import torch

from s2fft.base_transforms import wigner as base
from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms.wigner import _kernel_functions, forward, inverse
from s2fft.sampling import so3_samples as samples

jax.config.update("jax_enable_x64", True)

L_to_test = [6]
N_to_test = [2, 6]
nside_to_test = [4]
L_to_nside_ratio = [2]
reality_to_test = [False, True]
sampling_schemes = ["mw", "mwss", "dh", "gl"]
methods_to_test = ["numpy", "jax", "torch"]
modes_to_test = ["auto", "fft", "direct"]


def check_mode_and_sampling(mode, sampling):
    if mode.lower() == "fft" and sampling.lower() not in ["mw", "mwss", "dh"]:
        pytest.skip(
            f"Fourier based Wigner computation not valid for sampling={sampling}"
        )


def get_flmn_and_kernel(
    flmn_generator, L, N, sampling, reality, method, mode, forward, nside=None
):
    flmn = flmn_generator(L=L, N=N, reality=reality)
    kfunc = _kernel_functions[method]
    kernel = kfunc(L, N, reality, sampling, nside, forward=forward, mode=mode)
    return flmn, kernel


def check_inverse_transform(flmn, kernel, L, N, sampling, reality, method, nside=None):
    f = inverse(
        torch.from_numpy(flmn) if method == "torch" else flmn,
        L,
        N,
        kernel,
        sampling,
        reality,
        method,
        nside,
    )
    if method == "torch":
        f = f.resolve_conj().numpy()
    f_check = base.inverse(flmn, L, N, 0, sampling, reality, nside)
    np.testing.assert_allclose(f, f_check, atol=1e-5, rtol=1e-5)


def check_forward_tranform(flmn, kernel, L, N, sampling, reality, method, nside=None):
    f = base.inverse(flmn, L, N, sampling=sampling, reality=reality, nside=nside)
    flmn_check = base.forward(f, L, N, sampling=sampling, reality=reality, nside=nside)
    flmn = forward(
        torch.from_numpy(f) if method == "torch" else f,
        L,
        N,
        kernel,
        sampling,
        reality,
        method,
        nside,
    )
    if method == "torch":
        flmn = flmn.resolve_conj().numpy()
    np.testing.assert_allclose(flmn, flmn_check, atol=1e-5, rtol=1e-5)


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
    check_mode_and_sampling(mode, sampling)
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=False
    )
    check_inverse_transform(flmn, kernel, L, N, sampling, reality, method)


@pytest.mark.slow
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("mode", modes_to_test)
def test_inverse_wigner_transform_torch_gradcheck(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    mode: str,
):
    method = "torch"
    check_mode_and_sampling(mode, sampling)
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=False
    )
    flmn = torch.from_numpy(flmn)
    flmn.requires_grad = True
    assert torch.autograd.gradcheck(
        inverse, (flmn, L, N, kernel, sampling, reality, method)
    )


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
    check_mode_and_sampling(mode, sampling)
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=True
    )
    check_forward_tranform(flmn, kernel, L, N, sampling, reality, method)


@pytest.mark.slow
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("mode", modes_to_test)
def test_forward_wigner_transform_torch_gradcheck(
    flmn_generator,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    mode: str,
):
    method = "torch"
    check_mode_and_sampling(mode, sampling)
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=True
    )
    f = base.inverse(flmn, L, N, sampling=sampling, reality=reality)
    f = torch.from_numpy(f)
    f.requires_grad = True
    assert torch.autograd.gradcheck(
        forward, (f, L, N, kernel, sampling, reality, method)
    )


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
    mode = "auto"
    L = ratio * nside
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator,
        L,
        N,
        sampling,
        reality,
        method,
        mode,
        forward=False,
        nside=nside,
    )
    check_inverse_transform(flmn, kernel, L, N, sampling, reality, method, nside)


@pytest.mark.slow
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_wigner_transform_healpix_torch_gradcheck(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
):
    method = "torch"
    sampling = "healpix"
    mode = "auto"
    L = ratio * nside
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator,
        L,
        N,
        sampling,
        reality,
        method,
        mode,
        forward=False,
        nside=nside,
    )
    flmn = torch.from_numpy(flmn)
    flmn.requires_grad = True
    assert torch.autograd.gradcheck(
        inverse, (flmn, L, N, kernel, sampling, reality, method, nside)
    )


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
    mode = "auto"
    L = ratio * nside
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=True, nside=nside
    )
    check_forward_tranform(flmn, kernel, L, N, sampling, reality, method, nside)


@pytest.mark.slow
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_wigner_transform_healpix_torch_gradcheck(
    flmn_generator,
    nside: int,
    ratio: int,
    N: int,
    reality: bool,
):
    method = "torch"
    sampling = "healpix"
    mode = "auto"
    L = ratio * nside
    flmn, kernel = get_flmn_and_kernel(
        flmn_generator, L, N, sampling, reality, method, mode, forward=True, nside=nside
    )
    f = base.inverse(flmn, L, N, sampling=sampling, reality=reality, nside=nside)
    f = torch.from_numpy(f)
    f.requires_grad = True
    assert torch.autograd.gradcheck(
        forward, (f, L, N, kernel, sampling, reality, method, nside)
    )


@pytest.mark.parametrize("L", [8, 16, 32, 64])
@pytest.mark.parametrize("fft_method", [True, False])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_inverse_wigner_transform_high_N(
    cached_so3_test_case: Callable,
    L: int,
    fft_method: bool,
    sampling: str,
    reality: bool,
):
    if sampling.lower() in ["gl", "dh"]:
        pytest.skip("SO3 benchmark only supports [mw, mwss] sampling.")

    N = int(L / np.log(L)) if fft_method else L

    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    kernel = c.wigner_kernel_jax(L, N, reality, sampling, forward=False)
    f_check = inverse(test_data["flmn"], L, N, kernel, sampling, reality, "numpy")

    np.testing.assert_allclose(
        test_data["f_so3"], f_check.flatten("C"), atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("L", [8, 16, 32, 64])
@pytest.mark.parametrize("fft_method", [True, False])
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("reality", reality_to_test)
def test_forward_wigner_transform_high_N(
    cached_so3_test_case: Callable,
    L: int,
    fft_method: bool,
    sampling: str,
    reality: bool,
):
    if sampling.lower() in ["gl", "dh"]:
        pytest.skip("SO3 benchmark only supports [mw, mwss] sampling.")

    N = int(L / np.log(L)) if fft_method else L

    test_data = cached_so3_test_case(
        L=L, N=N, L_lower=0, sampling=sampling, reality=reality
    )

    f = test_data["f_so3"].reshape(
        samples._ngamma(N),
        samples._nbeta(L, sampling),
        samples._nalpha(L, sampling),
    )

    kernel = c.wigner_kernel(L, N, reality, sampling, forward=True)
    flmn_check = forward(f, L, N, kernel, sampling, reality, "numpy")

    np.testing.assert_allclose(
        test_data["flmn_so3"], flmn_check, atol=1e-10, rtol=1e-10
    )
