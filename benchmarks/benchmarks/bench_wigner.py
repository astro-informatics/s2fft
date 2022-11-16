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
