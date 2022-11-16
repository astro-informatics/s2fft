# set threads used by numpy (before numpy is imported!)
# import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np

from functools import partial
import timeit as timeit
import importlib

import jax.numpy as jnp
import jax as jax
from jax.config import config

config.update("jax_enable_x64", True)  # this only works on startup!

import s2fft.wigner as wigner
import s2fft.samples as samples
import pyssht as ssht

# summary of available compute resources
var = {}
var["jax_devices"] = jax.devices()
var["jax_local_devices"] = jax.local_devices()
var["jax_devices_cpu"] = jax.devices("cpu")
var["jax_device_count_cpu"] = jax.device_count("cpu")
# var['jax_devices_gpu'] = jax.devices('gpu')
# var['jax_device_count_gpu'] = jax.device_count('gpu')

# list of different parameters to benchmark
par = {}
# recursions
# par["RECURSION"] = ["risbo", "trapani", "turok"]
par["RECURSION"] = ["turok"]
# data types
# par["DATATYPE"] = ["float32", "float64"]
par["DATATYPE"] = ["float64"]
# implementations
par["IMPLEMENTATION"] = ["loop", "vectorized", "jax"]
# harmonic band-limit
# par["BANDLIMIT"] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
par["BANDLIMIT"] = [32, 64]
# sampling schemes
# par["SAMPLING"] = ["mw", "mwss", "dh", "healpix"]
# colatitude
par["COLATITUDE"] = [np.pi / 2.0]


T = {}
for count_recursion in par["RECURSION"]:
    for count_datatype in par["DATATYPE"]:
        for count_implementation in par["IMPLEMENTATION"]:
            for count_colatitude in par["COLATITUDE"]:
                for count_bandlimit in par["BANDLIMIT"]:
                    func = partial(
                        func_benchmark_runtime,
                        count_recursion,
                        count_implementation,
                        count_bandlimit,
                        count_colatitude,
                        count_datatype,
                    )
                    tempT = timeit.timeit(func, number=10)
                    if count_implementation in ["jax"]:
                        tempT = timeit.timeit(func, number=10)
                    # T[count_recursion][str(count_datatype)] = tempT
                    print(
                        "{:7}".format(count_recursion),
                        "{:10}".format(count_implementation),
                        "T(" + "{:09.4f}".format(tempT) + "s) "
                        "L(" + "{:04d}".format(count_bandlimit) + ") ",
                        "beta("
                        + "{:03.4f}".format(count_colatitude * 180 / (2 * np.pi))
                        + ") ",
                        count_datatype,
                    )
