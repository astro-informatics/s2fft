"""Benchmarks for on-the-fly spherical transforms."""

import numpy as np
from benchmarking import (
    BenchmarkSetup,
    benchmark,
    parse_args_collect_and_run_benchmarks,
)

import s2fft

L_VALUES = [8, 16, 32, 64, 128, 256]
L_LOWER_VALUES = [0]
SPIN_VALUES = [0]
L_TO_NSIDE_RATIO_VALUES = [2]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
N_ITER_VALUES = [None]


def _get_nside(sampling, L, L_to_nside_ratio):
    return None if sampling != "healpix" else L // L_to_nside_ratio


def setup_forward(
    method, L, L_lower, sampling, spin, L_to_nside_ratio, reality, n_iter
):
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    nside = _get_nside(sampling, L, L_to_nside_ratio)
    f = s2fft.transforms.spherical.inverse(
        flm,
        L=L,
        spin=spin,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
        L_lower=L_lower,
    )
    return BenchmarkSetup({"f": f}, flm, "jax" in method)


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    L_to_nside_ratio=L_TO_NSIDE_RATIO_VALUES,
    reality=REALITY_VALUES,
    n_iter=N_ITER_VALUES,
)
def forward(
    f,
    method,
    L,
    L_lower,
    sampling,
    spin,
    L_to_nside_ratio,
    reality,
    n_iter,
):
    return s2fft.transforms.spherical.forward(
        f=f,
        L=L,
        L_lower=L_lower,
        spin=spin,
        nside=_get_nside(sampling, L, L_to_nside_ratio),
        sampling=sampling,
        reality=reality,
        method=method,
        iter=n_iter,
    )


def setup_inverse(method, L, L_lower, sampling, spin, L_to_nside_ratio, reality):
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    return BenchmarkSetup({"flm": flm}, None, "jax" in method)


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    L_to_nside_ratio=L_TO_NSIDE_RATIO_VALUES,
    reality=REALITY_VALUES,
)
def inverse(flm, method, L, L_lower, sampling, spin, L_to_nside_ratio, reality):
    return s2fft.transforms.spherical.inverse(
        flm=flm,
        L=L,
        L_lower=L_lower,
        spin=spin,
        nside=_get_nside(sampling, L, L_to_nside_ratio),
        sampling=sampling,
        reality=reality,
        method=method,
    )


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
