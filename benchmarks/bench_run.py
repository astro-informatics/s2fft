"""
Benchmarks for Wigner-d recursions
"""

# set threads used by numpy (before numpy is imported!)
# import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


import argparse
import numpy as np

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

from utils import parametrize, parameters_string, run_benchmarks

from wigner import (
    risbo_compute_full_sequential,
    trapani_compute_full_sequential,
    trapani_compute_full_vectorized,
    trapani_compute_full_jax,
    turok_compute_full_sequential,
    turok_compute_full_largest_plane,
    turok_compute_slice_largest_plane,
    turok_jax_compute_slice_largest_plane,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Wigner recursion benchmarks")
    parser.add_argument("--number-runs", type=int, default=10)
    parser.add_argument("--number-repeats", type=int, default=3)
    args = parser.parse_args()

    results = run_benchmarks(
        benchmarks=[
            risbo_compute_full_sequential,
            trapani_compute_full_sequential,
            trapani_compute_full_vectorized,
            trapani_compute_full_jax,
            turok_compute_full_sequential,
            turok_compute_full_largest_plane,
            turok_compute_slice_largest_plane,
            turok_jax_compute_slice_largest_plane,
        ],
        number_runs=args.number_runs,
        number_repeats=args.number_repeats,
    )
