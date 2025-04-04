import jax
import numpy as np
import pyssht as ssht
import pytest
import torch

from s2fft.base_transforms import spherical as base
from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms.spherical import forward, inverse
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
iter_to_test = [0, 3]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
@pytest.mark.filterwarnings("ignore:Casting complex values")
def test_transform_inverse(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
    recursion: str,
):
    if recursion.lower() == "price-mcewen" and abs(spin) >= PM_MAX_STABLE_SPIN:
        pytest.skip(
            f"price-mcewen recursion not accurate above |spin| = {PM_MAX_STABLE_SPIN}"
        )

    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_check = base.inverse(flm, L, spin, sampling, reality=reality)

    kfunc = (
        c.spin_spherical_kernel_jax
        if method.lower() == "jax"
        else c.spin_spherical_kernel
    )
    kernel = kfunc(L, spin, reality, sampling, forward=False, recursion=recursion)

    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12
    if method.lower() == "torch":
        f = inverse(
            torch.from_numpy(flm),
            L,
            spin,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
        )
        # Test Transform
        np.testing.assert_allclose(
            f.resolve_conj().numpy(), f_check, atol=tol, rtol=tol
        )
        # Test Gradients
        flm_grad_test = torch.from_numpy(flm)
        flm_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            inverse,
            (
                flm_grad_test,
                L,
                spin,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
            ),
        )

    else:
        f = inverse(flm, L, spin, kernel, sampling, reality, method)
        np.testing.assert_allclose(f, f_check, atol=tol, rtol=tol)


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
    L = ratio * nside
    flm = flm_generator(L=L, reality=reality)
    f_check = base.inverse(flm, L, 0, sampling, nside, reality)

    kfunc = (
        c.spin_spherical_kernel_jax
        if method.lower() == "jax"
        else c.spin_spherical_kernel
    )
    kernel = kfunc(L, 0, reality, sampling, nside, False, recursion=recursion)

    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12
    if method.lower() == "torch":
        # Test Transform
        f = inverse(
            torch.from_numpy(flm),
            L,
            0,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
            nside,
        )
        np.testing.assert_allclose(f, f_check, atol=tol, rtol=tol)

        # Test Gradients
        flm_grad_test = torch.from_numpy(flm)
        flm_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            inverse,
            (
                flm_grad_test,
                L,
                0,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
                nside,
            ),
        )
    else:
        f = inverse(flm, L, 0, kernel, sampling, reality, method, nside)
        np.testing.assert_allclose(f, f_check, atol=tol, rtol=tol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("method", methods_to_test)
@pytest.mark.parametrize("recursion", recursions_to_test)
@pytest.mark.filterwarnings("ignore:Casting complex values")
def test_transform_forward(
    flm_generator,
    L: int,
    spin: int,
    sampling: str,
    reality: bool,
    method: str,
    recursion: str,
):
    if recursion.lower() == "price-mcewen" and abs(spin) >= PM_MAX_STABLE_SPIN:
        pytest.skip(
            f"price-mcewen recursion not accurate above |spin| = {PM_MAX_STABLE_SPIN}"
        )

    flm = flm_generator(L=L, spin=spin, reality=reality)

    f = base.inverse(flm, L, spin, sampling, reality=reality)
    flm_check = base.forward(f, L, spin, sampling, reality=reality)

    kfunc = (
        c.spin_spherical_kernel_jax
        if method.lower() == "jax"
        else c.spin_spherical_kernel
    )
    kernel = kfunc(L, spin, reality, sampling, forward=True, recursion=recursion)

    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12
    if method.lower() == "torch":
        # Test Transform
        flm_recov = forward(
            torch.from_numpy(f),
            L,
            spin,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
        )

        np.testing.assert_allclose(flm_check, flm_recov, atol=tol, rtol=tol)

        # Test Gradients
        f_grad_test = torch.from_numpy(f)
        f_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            forward,
            (
                f_grad_test,
                L,
                spin,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
            ),
        )
    else:
        flm_recov = forward(f, L, spin, kernel, sampling, reality, method)
        np.testing.assert_allclose(flm_check, flm_recov, atol=tol, rtol=tol)


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
    L = ratio * nside
    flm = flm_generator(L=L, reality=True)
    f = base.inverse(flm, L, 0, sampling, nside, reality)
    flm_check = base.forward(f, L, 0, sampling, nside, reality, iter=iter)

    kfunc = (
        c.spin_spherical_kernel_jax
        if method.lower() == "jax"
        else c.spin_spherical_kernel
    )
    kernel = kfunc(L, 0, reality, sampling, nside, True, recursion=recursion)

    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12
    if method.lower() == "torch":
        # Test Transform
        flm_recov = forward(
            torch.from_numpy(f),
            L,
            0,
            torch.from_numpy(kernel),
            sampling,
            reality,
            method,
            nside,
            iter,
        )
        np.testing.assert_allclose(flm_recov, flm_check, atol=tol, rtol=tol)

        # Test Gradients
        f_grad_test = torch.from_numpy(f)
        f_grad_test.requires_grad = True
        assert torch.autograd.gradcheck(
            forward,
            (
                f_grad_test,
                L,
                0,
                torch.from_numpy(kernel),
                sampling,
                reality,
                method,
                nside,
                iter,
            ),
        )
    else:
        flm_recov = forward(f, L, 0, kernel, sampling, reality, method, nside, iter)
        np.testing.assert_allclose(flm_recov, flm_check, atol=tol, rtol=tol)


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
    tol = 1e-8 if sampling.lower() in ["dh", "gl"] else 1e-12
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
