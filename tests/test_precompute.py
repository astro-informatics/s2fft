import os
import numpy as np
import jax.numpy as jnp
from jax import jit, device_put
import s2fft as s2f
import s2fft.precompute as pre
import pyssht as ssht
import pytest


from jax.config import config

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
def test_forward_legendre_matrix_constructor(L: int, spin: int):
    """Test creation and saving down of forward Legendre kernels"""
    sampling_method = "mw"
    save_dir = ".matrices"
    filename = save_dir + "/legendre_matrix_{}_{}_spin_{}.npy".format(
        L, sampling_method, spin
    )

    legendre_forward = pre.construct_legendre_matrix.construct_legendre_matrix(
        L, sampling_method, save_dir, spin
    )
    assert legendre_forward.shape == (L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    legendre_forward = pre.construct_legendre_matrix.construct_legendre_matrix(
        L, sampling_method, save_dir, spin
    )


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
def test_inverse_legendre_matrix_constructor(L: int, spin: int):
    """Test creation and saving down of inverse Legendre kernels"""
    sampling_method = "mw"
    save_dir = ".matrices"
    filename = save_dir + "/legendre_inverse_matrix_{}_{}_spin_{}.npy".format(
        L, sampling_method, spin
    )

    legendre_inverse = pre.construct_legendre_matrix.construct_legendre_matrix_inverse(
        L, sampling_method, save_dir, spin
    )
    assert legendre_inverse.shape == (L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    legendre_inverse = pre.construct_legendre_matrix.construct_legendre_matrix_inverse(
        L, sampling_method, save_dir, spin
    )


def test_legendre_matrix_constructor_compute_time_warning():
    """Test compile time warning of legendre kernels"""
    pre.construct_legendre_matrix.compute_warning(256)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_transform_precompute(flm_generator, L: int, spin: int, device: str):
    """Test wrapper implementation of forward/inverse precompute sht"""
    sampling_method = "mw"
    save_dir = ".matrices"

    leg_for = pre.construct_legendre_matrix.construct_legendre_matrix(
        L, sampling_method, save_dir, spin
    )
    leg_inv = pre.construct_legendre_matrix.construct_legendre_matrix_inverse(
        L, sampling_method, save_dir, spin
    )

    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(s2f.samples.flm_2d_to_1d(flm, L), L, spin)

    flm_precomp = pre.transforms.forward_precompute(f, L, leg_for, device, spin)
    assert np.allclose(flm_precomp, flm)

    f_precomp = pre.transforms.inverse_precompute(flm_precomp, L, leg_inv, device, spin)
    assert np.allclose(f_precomp, f)


def test_transform_device_exceptions():
    """Test device exception of forward/inverse precompute sht"""
    f = ["not a vector"]
    legendre_kernel = ["not a kernel"]
    device = ["not a device"]
    with pytest.raises(ValueError):
        pre.transforms.forward_precompute(f, 8, legendre_kernel, device, 0)
    with pytest.raises(ValueError):
        pre.transforms.inverse_precompute(f, 8, legendre_kernel, device, 0)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_transform_precompute_load_legendre(
    flm_generator, L: int, spin: int, device: str
):
    """Test wrapper implementation of forward/inverse precompute sht"""
    sampling_method = "mw"
    save_dir = ".matrices"

    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(s2f.samples.flm_2d_to_1d(flm, L), L, spin)

    flm_precomp = pre.transforms.forward_precompute(
        f, L=L, device=device, spin=spin, save_dir=save_dir
    )
    assert np.allclose(flm_precomp, flm)

    f_precomp = pre.transforms.inverse_precompute(
        flm_precomp, L=L, device=device, spin=spin, save_dir=save_dir
    )
    assert np.allclose(f_precomp, f)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
def test_transform_precompute_cpu(flm_generator, L: int, spin: int):
    """Test cpu implementation of forward/inverse precompute sht"""
    sampling_method = "mw"
    save_dir = ".matrices"

    leg_for = pre.construct_legendre_matrix.construct_legendre_matrix(
        L, sampling_method, save_dir, spin
    )
    leg_inv = pre.construct_legendre_matrix.construct_legendre_matrix_inverse(
        L, sampling_method, save_dir, spin
    )

    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(s2f.samples.flm_2d_to_1d(flm, L), L, spin)

    flm_cpu = pre.transforms.forward_transform_cpu(f, leg_for, L)
    assert np.allclose(flm_cpu, flm)

    f_cpu = pre.transforms.inverse_transform_cpu(flm_cpu, leg_inv, L)
    assert np.allclose(f_cpu, f)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
def test_transform_precompute_gpu(flm_generator, L: int, spin: int):
    """Test gpu implementation of forward/inverse precompute sht"""
    sampling_method = "mw"
    save_dir = ".matrices"

    leg_for = device_put(
        pre.construct_legendre_matrix.construct_legendre_matrix(
            L, sampling_method, save_dir, spin
        )
    )
    leg_inv = device_put(
        pre.construct_legendre_matrix.construct_legendre_matrix_inverse(
            L, sampling_method, save_dir, spin
        )
    )

    flm = flm_generator(L=L, spin=spin, reality=False)

    f = ssht.inverse(s2f.samples.flm_2d_to_1d(flm, L), L, spin)

    forward_jit = jit(pre.transforms.forward_transform_gpu, static_argnums=(2,))
    inverse_jit = jit(pre.transforms.inverse_transform_gpu, static_argnums=(2,))

    flm_gpu = forward_jit(device_put(f), leg_for, L)
    flm_gpu = np.array(flm_gpu)
    assert np.allclose(flm_gpu, flm)

    f_gpu = inverse_jit(device_put(flm_gpu), leg_inv, L)
    assert np.allclose(f_gpu, f)
