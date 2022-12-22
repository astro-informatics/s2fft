"""
Benchmarks for Wigner-d recursions
"""



import argparse
import numpy as np
import jax
import jax.numpy as jnp

import pyssht as ssht
import s2fft

from jax.config import config

config.update("jax_enable_x64", True)  # this only works on startup!

from utils import parametrize, parameters_string, run_benchmarks, print_summary

# list of different parameters to benchmark
# harmonic band-limit
L_VALUES = [2, 8]
# colatitude
BETA = np.pi / 2
# harmonic order
MM = 0


# Risbo
@parametrize({"L": L_VALUES})
def risbo_compute_full_sequential(L):
    """
    Risbo: compute the wigner-d plane for L - 1 by calculating the wigner-d planes for all 0 <= el <= L - 1
    """
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    for el in range(0, L):
        dl = s2fft.wigner.risbo.compute_full(dl, np.pi / 2, L, el)


# Trapani
@parametrize({"L": L_VALUES})
def trapani_compute_full_sequential(L):
    """
    Trapani: compute the wigner-d plane for L - 1 by calculating the wigner-d planes for all 0 <= el <= L - 1
    """
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    dl = s2fft.wigner.trapani.init(dl, L, "loop")
    for el in range(1, L):
        dl = s2fft.wigner.trapani.compute_full(dl, L, el, "loop")


@parametrize({"L": L_VALUES})
def trapani_compute_full_vectorized(L):
    """
    Trapani: compute the wigner-d plane for L - 1 by calculating the wigner-d planes for all 0 <= el <= L - 1 (vectorized)
    """
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    dl = s2fft.wigner.trapani.init(dl, L, "vectorized")
    for el in range(1, L):
        dl = s2fft.wigner.trapani.compute_full(dl, L, el, "vectorized")


@parametrize({"L": L_VALUES})
def trapani_compute_full_jax(L):
    """
    Trapani: compute the wigner-d plane for L - 1 by calculating the wigner-d planes for all 0 <= el <= L - 1 (jax)
    """
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    dl = s2fft.wigner.trapani.init(dl, L, "jax")
    for el in range(1, L):
        dl = s2fft.wigner.trapani.compute_full(dl, L, el, "jax").block_until_ready()


# Turok
@parametrize({"L": L_VALUES})
def turok_compute_full_sequential(L):
    """
    Turok: compute wigner-d planes for all 0 <= el <= L - 1. Only for comparison purposes.
    """
    for el in range(0, L):
        dl = s2fft.wigner.turok.compute_full(BETA, el, L)


@parametrize({"L": L_VALUES})
def turok_compute_full_largest_plane(L):
    """
    Turok: compute wigner-d plane for L - 1
    """
    dl = s2fft.wigner.turok.compute_full(BETA, L - 1, L)


@parametrize({"L": L_VALUES})
def turok_compute_slice_largest_plane(L):
    """
    Turok: compute slice of wigner-d plane for L - 1 at MM
    """
    dl = s2fft.wigner.turok.compute_slice(BETA, L - 1, L, MM)


@parametrize({"L": L_VALUES})
def turok_jax_compute_slice_largest_plane(L):
    """
    Turok: compute slice of wigner-d plane for L - 1 at MM (jax)
    """
    dl = s2fft.wigner.turok_jax.compute_slice(BETA, L - 1, L, MM).block_until_ready()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Wigner recursion benchmarks")
    parser.add_argument(
        "--number-runs",
        type=int,
        default=10,
        help="number of times the script is timed",
    )
    parser.add_argument(
        "--number-repeats",
        type=int,
        default=3,
        help="number of times the timer is repeated",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        default=False,
        help="print basic summary",
    )

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
    if args.print_summary:
        print_summary(results)
