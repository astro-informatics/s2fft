"""Benchmarks for Wigner-d recursions"""

# set threads used by numpy (before numpy is imported!)
# import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

from functools import partial
from itertools import product
import argparse
import numpy as np
import timeit
import pyssht as ssht
import s2fft
import jax
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)  # this only works on startup!

# summary of available compute resources
# var = {}
# var["jax_devices"] = jax.devices()
# var["jax_local_devices"] = jax.local_devices()
# var["jax_devices_cpu"] = jax.devices("cpu")
# var["jax_device_count_cpu"] = jax.device_count("cpu")
# var['jax_devices_gpu'] = jax.devices('gpu')
# var['jax_device_count_gpu'] = jax.device_count('gpu')

# list of different parameters to benchmark
# harmonic band-limit
L_VALUES = [16, 32]
# colatitude
BETA = np.pi / 2
# harmonic order
MM = 0


def parametrize(parameter_dict):
    """
    Returns a function that unpacks a dictionary's keys and values
    """

    def decorator(function):
        function.param_names = list(parameter_dict.keys())
        function.params = list(parameter_dict.values())
        return function

    return decorator


@parametrize({"L": L_VALUES})
def risbo_compute_full_sequential(L):
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    for el in range(0, L):
        s2fft.wigner.risbo.compute_full(dl, BETA, L, el)


@parametrize({"L": L_VALUES, "implementation": ("loop", "vectorized", "jax")})
def trapani_compute_full_sequential(L, implementation):
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    dl = s2fft.wigner.trapani.init(dl, L, implementation)
    for el in range(1, L):
        dl = s2fft.wigner.trapani.compute_full(dl, L, el, implementation)
    if implementation == "jax":
        dl = dl.block_until_ready()


@parametrize({"L": L_VALUES})
def turok_compute_full_sequential(L):
    for el in range(L):
        s2fft.wigner.turok.compute_full(BETA, el, L)


@parametrize({"L": L_VALUES})
def turok_compute_full_largest_plane(L):
    s2fft.wigner.turok.compute_full(BETA, L - 1, L)


@parametrize({"L": L_VALUES})
def turok_compute_slice_largest_plane(L):
    s2fft.wigner.turok.compute_slice(BETA, L - 1, L, MM)


@parametrize({"L": L_VALUES})
def turok_jax_compute_slice_largest_plane(L):
    s2fft.wigner.turok_jax.compute_slice(BETA, L - 1, L, MM).block_until_ready()
