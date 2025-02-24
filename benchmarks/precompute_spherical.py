"""Benchmarks for precompute spherical transforms."""

import numpy as np
from benchmarking import (
    BenchmarkSetup,
    benchmark,
    parse_args_collect_and_run_benchmarks,
    skip,
)

import s2fft
import s2fft.precompute_transforms

L_VALUES = [8, 16, 32, 64, 128, 256]
SPIN_VALUES = [0]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
RECURSION_VALUES = ["auto"]


def setup_forward(method, L, sampling, spin, reality, recursion):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    f = s2fft.transforms.spherical.inverse(
        flm,
        L=L,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    if method == "torch":
        import torch

        flm = torch.from_numpy(flm)
        f = torch.from_numpy(f)
    kernel_function = s2fft.precompute_transforms.spherical._kernel_functions[method]
    kernel = kernel_function(
        L=L,
        spin=spin,
        reality=reality,
        sampling=sampling,
        forward=True,
        recursion=recursion,
    )
    return BenchmarkSetup({"f": f, "kernel": kernel}, flm, "jax" in method)


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    reality=REALITY_VALUES,
    recursion=RECURSION_VALUES,
)
def forward(f, kernel, method, L, sampling, spin, reality, recursion):
    return s2fft.precompute_transforms.spherical.forward(
        f=f,
        L=L,
        spin=spin,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )


def setup_inverse(method, L, sampling, spin, reality, recursion):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    if method == "torch":
        import torch

        flm = torch.from_numpy(flm)
    kernel_function = s2fft.precompute_transforms.spherical._kernel_functions[method]
    kernel = kernel_function(
        L=L,
        spin=spin,
        reality=reality,
        sampling=sampling,
        forward=False,
        recursion=recursion,
    )
    return BenchmarkSetup({"flm": flm, "kernel": kernel}, None, "jax" in method)


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    reality=REALITY_VALUES,
    recursion=RECURSION_VALUES,
)
def inverse(flm, kernel, method, L, sampling, spin, reality, recursion):
    return s2fft.precompute_transforms.spherical.inverse(
        flm=flm,
        L=L,
        spin=spin,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
