import os 
import numpy as np 
import jax.numpy as jnp 
from jax import jit, device_put
import s2fft.precompute as pre
import pyssht as ssht 

from jax.config import config 

config.update("jax_enable_x64", True)

def test_forward_legendre_matrix_constructor():
    """Test creation and saving down of forward Legendre kernels"""
    L = 32 
    spin = 2
    sampling_method = "MW"
    save_dir = ".matrices"
    filename = save_dir + "/ssht_legendre_matrix_{}_{}_spin_{}.npy".format(L, sampling_method, spin)

    legendre_forward = pre.construct_legendre_matrix.construct_ssht_legendre_matrix(L, sampling_method, save_dir, spin)
    assert (legendre_forward.shape == (L, 2*L-1, L))
    assert (os.path.isfile(filename))
    legendre_forward = pre.construct_legendre_matrix.construct_ssht_legendre_matrix(L, sampling_method, save_dir, spin)

def test_inverse_legendre_matrix_constructor():
    """Test creation and saving down of inverse Legendre kernels"""
    L = 32 
    spin = 2
    sampling_method = "MW"
    save_dir = ".matrices"
    filename = save_dir + "/ssht_legendre_inverse_matrix_{}_{}_spin_{}.npy".format(L, sampling_method, spin)

    legendre_inverse = pre.construct_legendre_matrix.construct_ssht_legendre_matrix_inverse(L, sampling_method, save_dir, spin)
    assert (legendre_inverse.shape == (L, 2*L-1, L))
    assert (os.path.isfile(filename))
    legendre_inverse = pre.construct_legendre_matrix.construct_ssht_legendre_matrix_inverse(L, sampling_method, save_dir, spin)

def test_forward_transform_precompute_cpu():
    """Test cpu implementation of forward precompute sht"""
    sampling_method = "Mw"
    save_dir = ".matrices"

    for l in range(6):
        L = int(2**l)
        for spin in range(-5,5):
            leg_for = pre.construct_legendre_matrix.construct_ssht_legendre_matrix(L, sampling_method, save_dir, spin)
            leg_inv = pre.construct_legendre_matrix.construct_ssht_legendre_matrix_inverse(L, sampling_method, save_dir, spin)

            flm = np.random.randn(L*L) + 1j* np.random.randn(L*L)
            f = ssht.inverse(flm, L, spin)

            flm_cpu = pre.ssht_matrix.forward_ssht_transform_cpu(f, leg_for, L)
            assert np.allclose(flm_cpu[np.nonzero(flm_cpu)], flm[spin**2:])

            f_cpu = pre.ssht_matrix.inverse_ssht_transform_cpu(flm_cpu, leg_inv, L)
            assert np.allclose(f_cpu, f)

def test_forward_transform_precompute_gpu():
    """Test gpu implementation of forward precompute sht"""
    sampling_method = "Mw"
    save_dir = ".matrices"

    for l in range(6):
        L = int(2**l)
        for spin in range(-5,5):
            leg_for = device_put(pre.construct_legendre_matrix.construct_ssht_legendre_matrix(L, sampling_method, save_dir, spin))
            leg_inv = device_put(pre.construct_legendre_matrix.construct_ssht_legendre_matrix_inverse(L, sampling_method, save_dir, spin))

            flm = np.random.randn(L*L) + 1j* np.random.randn(L*L)
            f = ssht.inverse(flm, L, spin)

            forward_jit = jit(pre.ssht_matrix.forward_ssht_transform_gpu, static_argnums=(2,))
            inverse_jit = jit(pre.ssht_matrix.inverse_ssht_transform_gpu, static_argnums=(2,))

            flm_gpu = forward_jit(device_put(f), leg_for, L)
            flm_gpu = np.array(flm_gpu)
            assert np.allclose(flm_gpu[np.nonzero(flm_gpu)], flm[spin**2:])

            f_gpu = inverse_jit(device_put(flm_gpu), leg_inv, L)
            assert np.allclose(f_gpu, f)

                


