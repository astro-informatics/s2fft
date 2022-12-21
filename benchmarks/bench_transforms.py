"""
Benchmarks for transforms
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
import jax
import jax.numpy as jnp

import pyssht
import s2fft

from jax.config import config

config.update("jax_enable_x64", True)  # this only works on startup!

from utils import parametrize, run_benchmarks

# list of different parameters to benchmark
# harmonic band-limit
L_VALUES = [16]
# spin
SPIN = 0
# sampling scheme
SAMPLING_VALUES = ["mw"]
# implementation
METHOD_VALUES = [
    "pyssht",
]  # , "sov_fft_jax_map", "sov_fft_jax_scan", "sov_fft_jax_vmap"]

RNG = np.random.default_rng(1234)
F_ARRAYS = {
    (L, sampling): pyssht.inverse(
        s2fft.samples.flm_2d_to_1d(
            s2fft.utils.generate_flm(RNG, L, spin=SPIN, reality=True), L
        ),
        L,
        Method=sampling.upper(),
        Spin=SPIN,
        Reality=False,
    )
    for L in L_VALUES
    for sampling in SAMPLING_VALUES
}


@parametrize({"method": METHOD_VALUES, "L": L_VALUES, "sampling": SAMPLING_VALUES})
def forward_transform(method, L, sampling):
    f = F_ARRAYS[L, sampling]
    if method == "pyssht":
        flm = pyssht.forward(f, L, SPIN, sampling.upper())
    else:
        flm = s2fft.transform._forward(
            f, L, spin=SPIN, sampling=sampling, method=method
        )
        if "jax" in method:
            flm.block_until_ready()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run transform benchmarks")
    parser.add_argument("--number-runs", type=int, default=10)
    parser.add_argument("--number-repeats", type=int, default=3)
    args = parser.parse_args()

    results = run_benchmarks(
        benchmarks=[
            forward_transform,
        ],
        number_runs=args.number_runs,
        number_repeats=args.number_repeats,
    )
