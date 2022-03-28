import os
import numpy as np
import pyssht as ssht
from construct_legendre_matrix import (
    construct_ssht_legendre_matrix,
    construct_ssht_legendre_matrix_inverse,
)

from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import s2fft.logs as lg

lg.setup_logging()


def ssht_forward_precompute(
    f, L=4, sampling_method="MW", save_dir="../../.matrices", device="cpu"
):
    lg.info_log("Running precompute forward ssht for L={} on cpu".format(L))

    filepath = "{}/ssht_legendre_matrix_{}_{}_spin_0.npy".format(
        save_dir, L, sampling_method
    )

    if not os.path.isfile(filepath):
        construct_ssht_legendre_matrix(
            L=L,
            sampling_method=sampling_method,
            save_dir=save_dir,
        )
    legendre_kernel = np.load(filepath)

    if device == "cpu":
        return __forward_ssht_transform_cpu(f, legendre_kernel, L)
    elif device == "gpu":
        return __forward_ssht_transform_gpu(f, legendre_kernel, L)
    else:
        raise ValueError("Device not recognised.")


def __forward_ssht_transform_cpu(f, legendre_kernel, L):
    fm = np.fft.fft(f)
    flm = np.einsum("lmi, im->lm", legendre_kernel, fm)
    return __square_to_flat(flm, L)

@jit
def __forward_ssht_transform_gpu(f, legendre_kernel, L):
    fm = jnp.fft.fft(f)
    flm = jnp.einsum("lmi, im->lm", legendre_kernel, fm)
    # return __square_to_flat(flm, L)
    return flm

def ssht_inverse_precompute(
    flm, L=4, sampling_method="MW", save_dir="../../.matrices", device="cpu"
):

    lg.info_log("Running precompute inverse ssht for L={} on cpu".format(L))

    filepath = "{}/ssht_legendre_inverse_matrix_{}_{}_spin_0.npy".format(
        save_dir, L, sampling_method
    )

    if not os.path.isfile(filepath):
        construct_ssht_legendre_matrix_inverse(
            L=L,
            sampling_method=sampling_method,
            save_dir=save_dir,
        )
    legendre_kernel = np.load(filepath)

    if device == "cpu":
        return __inverse_ssht_transform_cpu(flm, legendre_kernel, L)
    elif device == "gpu":
        return __inverse_ssht_transform_gpu(flm, legendre_kernel, L)
    else:
        raise ValueError("Device not recognised.")

def __inverse_ssht_transform_cpu(flm, legendre_kernel, L):
    flm_sq = __flat_to_square(flm, L)
    fm = np.einsum("lmi, lm->im", legendre_kernel, flm_sq)
    return fm.shape[1] * np.fft.ifft(fm)

@jit
def __inverse_ssht_transform_gpu(flm, legendre_kernel, L):
    # flm_sq = __flat_to_square(flm, L)
    fm = jnp.einsum("lmi, lm->im", legendre_kernel, flm)
    return (2*L-1) * jnp.fft.ifft(fm)


def __square_to_flat(flm_sq, L):
    flm = np.zeros(L**2, dtype=np.complex128)
    for el in range(L):
        for m in range(-el, el + 1):
            index = el**2 + el + m
            flm[index] = flm_sq[el, m]

    return flm


def __flat_to_square(flm, L):
    flm_sq = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for el in range(L):
        for m in range(-el, el + 1):
            i = el**2 + el + m
            flm_sq[el, m] = flm[i]

    return flm_sq


if __name__ == "__main__":
    L = 64
    save_dir="../../.matrices"
    sampling_method="MW"

    filepath_inv = "{}/ssht_legendre_inverse_matrix_{}_{}_spin_0.npy".format(
        save_dir, L, sampling_method
    )

    if not os.path.isfile(filepath_inv):
        construct_ssht_legendre_matrix_inverse(
            L=L,
            sampling_method=sampling_method,
            save_dir=save_dir,
        )
    legendre_kernel_inverse = np.load(filepath_inv)

    filepath_for = "{}/ssht_legendre_matrix_{}_{}_spin_0.npy".format(
        save_dir, L, sampling_method
    )

    if not os.path.isfile(filepath_for):
        construct_ssht_legendre_matrix(
            L=L,
            sampling_method=sampling_method,
            save_dir=save_dir,
        )
    legendre_kernel_forward = np.load(filepath_for)

    devices = ["cpu", "gpu"]

    f = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)
    f = ssht.inverse(ssht.forward(f, L), L)
    flm_true = ssht.forward(f, L=L)


    from timeit import default_timer as timer
    sub_iters = 10000
    for dev in devices:

        print("\nBenchmarking for device {}".format(dev))

        if dev == "cpu":
            start = timer()
            for i in range(sub_iters):
                flm_est = __forward_ssht_transform_cpu(f, legendre_kernel_forward, L)
            end = timer()
        elif dev == "gpu":
            forward_jit = jit(__forward_ssht_transform_gpu)
            start = timer()
            for i in range(sub_iters):
                flm_est = forward_jit(f, legendre_kernel_forward, L).block_until_ready()
                # flm_est = __forward_ssht_transform_gpu(f, legendre_kernel_forward, L)
            end = timer()

        print("    Device {} || Forward transform time: {}".format(dev, (end-start)/sub_iters))
        # print("    Device {} || Forward mean absolute error --> real = {}, imag = {}".format(dev, np.nanmean(np.abs(np.real(flm_est - flm_true))),np.nanmean(np.abs(np.imag(flm_est - flm_true)))))

        if dev == "cpu":
            start = timer()
            for i in range(sub_iters):
                f_est = __inverse_ssht_transform_cpu(flm_est, legendre_kernel_inverse, L)
            end = timer()
        elif dev == "gpu":
            inverse_jit = jit(__inverse_ssht_transform_gpu)
            start = timer()
            for i in range(sub_iters):
                f_est = inverse_jit(flm_est, legendre_kernel_inverse, L).block_until_ready()
                # f_est = __inverse_ssht_transform_gpu(flm_est, legendre_kernel_inverse, L)
            end = timer()

        print("    Device {} || Inverse transform time: {}".format(dev, (end-start)/sub_iters))
        print("    Device {} || Inverse mean absolute error --> real = {}, imag = {}\n".format(dev, np.nanmean(np.abs(np.real(f_est - f))),np.nanmean(np.abs(np.imag(f_est - f)))))
