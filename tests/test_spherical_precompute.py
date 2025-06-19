import jax
import numpy as np
import pyssht as ssht
import pytest
import torch

from s2fft.base_transforms import spherical as base
from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms.spherical import _kernel_functions, forward, inverse
from s2fft.sampling import s2_samples as samples

jax.config.update("jax_enable_x64", True)

# Maximum spin number at which Price-McEwen recursion is sufficiently accurate.
# For spins > PM_MAX_STABLE_SPIN one should default to the Risbo recursion.
PM_MAX_STABLE_SPIN = 6

L_to_test = [12]
spin_to_test = [-2, 0, 6]
nside_to_test = [4, 5]
L_to_nside_ratio = [2, 3]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
reality_to_test = [True, False]
methods_to_test = ["numpy", "jax", "torch"]
recursions_to_test = ["price-mcewen", "risbo", "auto"]
iter_to_test = [0, 1]


def get_flm_and_kernel(
    flm_generator,
    L,
    spin,
    sampling,
    reality,
    method,
    recursion,
    forward,
    nside=None,
):
    flm = flm_generator(L=L, spin=spin, reality=reality)
    kfunc = _kernel_functions[method]
    kernel = kfunc(L, spin, reality, sampling, nside, forward, recursion=recursion)
    return flm, kernel


def get_tol(sampling):
    return 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12


def check_spin(recursion, spin):
    if recursion.lower() == "price-mcewen" and abs(spin) >= PM_MAX_STABLE_SPIN:
        pytest.skip(
            f"price-mcewen recursion not accurate above |spin| = {PM_MAX_STABLE_SPIN}"
        )


def check_inverse_transform(
    flm, kernel, L, spin, sampling, reality, method, nside=None
):
    f = inverse(
        torch.from_numpy(flm) if method == "torch" else flm,
        L,
        spin,
        kernel,
        sampling,
        reality,
        method,
        nside,
    )
    if method == "torch":
        f = f.resolve_conj().numpy()
    f_check = base.inverse(flm, L, spin, sampling, nside, reality)
    tol = get_tol(sampling)
    np.testing.assert_allclose(f, f_check, atol=tol, rtol=tol)


def check_forward_transform(
    flm, kernel, L, spin, sampling, reality, method, nside=None
):
    f = base.inverse(flm, L, spin, sampling, nside, reality)
    flm_check = base.forward(f, L, spin, sampling, nside, reality)
    if method == "torch":
        f = torch.from_numpy(f)
    flm_recov = forward(f, L, spin, kernel, sampling, reality, method, nside)
    if method == "torch":
        flm_recov = flm_recov.resolve_conj().numpy()
    tol = get_tol(sampling)
    np.testing.assert_allclose(flm_check, flm_recov, atol=tol, rtol=tol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_inverse(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
    recursion: str,
):
    check_spin(recursion, spin)
    flm, kernel = get_flm_and_kernel(
        flm_generator, L, spin, sampling, reality, method, recursion, forward=False
    )
    check_inverse_transform(flm, kernel, L, spin, sampling, reality, method)


@pytest.mark.slow
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_inverse_torch_gradcheck(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    recursion: str,
):
    method = "torch"
    flm, kernel = get_flm_and_kernel(
        flm_generator, L, spin, sampling, reality, method, recursion, forward=False
    )
    flm = torch.from_numpy(flm)
    flm.requires_grad = True
    torch.autograd.gradcheck(inverse, (flm, L, spin, kernel, sampling, reality, method))


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_inverse_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    method: str,
    recursion: str,
):
    sampling = "healpix"
    spin = 0
    L = ratio * nside
    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=False,
        nside=nside,
    )
    check_inverse_transform(flm, kernel, L, spin, sampling, reality, method, nside)


@pytest.mark.slow
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_inverse_healpix_torch_gradcheck(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    recursion: str,
):
    method = "torch"
    sampling = "healpix"
    spin = 0
    L = ratio * nside
    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=False,
        nside=nside,
    )
    flm = torch.from_numpy(flm)
    flm.requires_grad = True
    torch.autograd.gradcheck(
        inverse, (flm, L, spin, kernel, sampling, reality, method, nside)
    )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_forward(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
    recursion: str,
):
    check_spin(recursion, spin)
    flm, kernel = get_flm_and_kernel(
        flm_generator, L, spin, sampling, reality, method, recursion, forward=True
    )
    check_forward_transform(flm, kernel, L, spin, sampling, reality, method)


@pytest.mark.slow
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
def test_transform_forward_torch_gradcheck(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    recursion: str,
):
    method = "torch"
    flm, kernel = get_flm_and_kernel(
        flm_generator, L, spin, sampling, reality, method, recursion, forward=True
    )
    f = torch.from_numpy(base.inverse(flm, L, spin, sampling, reality=reality))
    f.requires_grad = True
    torch.autograd.gradcheck(forward, (f, L, spin, kernel, sampling, reality, method))


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
@pytest.mark.parametrize("iter", iter_to_test)
def test_transform_forward_healpix(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    method: str,
    recursion: str,
    iter: int,
):
    sampling = "healpix"
    spin = 0
    L = ratio * nside
    check_spin(recursion, spin)
    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=True,
        nside=nside,
    )
    check_forward_transform(flm, kernel, L, spin, sampling, reality, method, nside)


@pytest.mark.slow
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("ratio", L_to_nside_ratio)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
@pytest.mark.parametrize("iter", iter_to_test)
def test_transform_forward_healpix_torch_gradcheck(
    flm_generator,
    nside: int,
    ratio: int,
    reality: bool,
    recursion: str,
    iter: int,
):
    method = "torch"
    sampling = "healpix"
    spin = 0
    L = ratio * nside
    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=True,
        nside=nside,
    )
    f = torch.from_numpy(base.inverse(flm, L, spin, sampling, nside, reality))
    f.requires_grad = True
    torch.autograd.gradcheck(
        forward, (f, L, spin, kernel, sampling, reality, method, nside)
    )


@pytest.mark.parametrize("spin", [0, 20, 30, -20, -30])
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_inverse_high_spin(
    flm_generator, spin: int, sampling: str, reality: bool
):
    L = 32

    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_check = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    kernel = c.spin_spherical_kernel(L, spin, reality, sampling, forward=False)

    f = inverse(flm, L, spin, kernel, sampling, reality, "numpy")
    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-11
    np.testing.assert_allclose(f, f_check, atol=tol, rtol=tol)


@pytest.mark.parametrize("spin", [0, 20, 30, -20, -30])
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_transform_forward_high_spin(
    flm_generator, spin: int, sampling: str, reality: bool
):
    L = 32

    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = ssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

    kernel = c.spin_spherical_kernel(L, spin, reality, sampling, forward=True)

    flm_recov = forward(f, L, spin, kernel, sampling, reality, "numpy")
    tol = get_tol(sampling)
    np.testing.assert_allclose(flm_recov, flm, atol=tol, rtol=tol)


def test_forward_transform_unrecognised_method_raises():
    method = "invalid_method"
    L = 32
    f = np.zeros(samples.f_shape(L))
    with pytest.raises(ValueError, match=f"{method} not recognised"):
        forward(f, L, method=method)


def test_inverse_transform_unrecognised_method_raises():
    method = "invalid_method"
    L = 32
    flm = np.zeros(samples.flm_shape(L))
    with pytest.raises(ValueError, match=f"{method} not recognised"):
        inverse(flm, L, method=method)
