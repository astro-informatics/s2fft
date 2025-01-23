"""Benchmarks for precompute Wigner-d transforms."""

import numpy as np
from benchmarking import benchmark, parse_args_collect_and_run_benchmarks

import s2fft
import s2fft.precompute_transforms
from s2fft.base_transforms import wigner as base_wigner

L_VALUES = [16, 32, 64, 128, 256]
N_VALUES = [2]
L_LOWER_VALUES = [0]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
MODE_VALUES = ["auto"]


def setup_forward(method, L, N, L_lower, sampling, reality, mode):
    rng = np.random.default_rng()
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L, N, reality=reality)
    f = base_wigner.inverse(
        flmn,
        L,
        N,
        L_lower=L_lower,
        sampling=sampling,
        reality=reality,
    )
    kernel_function = (
        s2fft.precompute_transforms.construct.wigner_kernel_jax
        if method == "jax"
        else s2fft.precompute_transforms.construct.wigner_kernel
    )
    kernel = kernel_function(
        L=L, N=N, reality=reality, sampling=sampling, forward=True, mode=mode
    )
    return {"f": f, "kernel": kernel}, flmn


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    N=N_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    reality=REALITY_VALUES,
    mode=MODE_VALUES,
)
def forward(f, kernel, method, L, N, L_lower, sampling, reality, mode):
    flmn = s2fft.precompute_transforms.wigner.forward(
        f=f,
        L=L,
        N=N,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    if method == "jax":
        flmn.block_until_ready()
    return flmn


def setup_inverse(method, L, N, L_lower, sampling, reality, mode):
    rng = np.random.default_rng()
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L, N, reality=reality)
    kernel_function = (
        s2fft.precompute_transforms.construct.wigner_kernel_jax
        if method == "jax"
        else s2fft.precompute_transforms.construct.wigner_kernel
    )
    kernel = kernel_function(
        L=L, N=N, reality=reality, sampling=sampling, forward=False, mode=mode
    )
    return {"flmn": flmn, "kernel": kernel}, None


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    N=N_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    reality=REALITY_VALUES,
    mode=MODE_VALUES,
)
def inverse(flmn, kernel, method, L, N, L_lower, sampling, reality, mode):
    f = s2fft.precompute_transforms.wigner.inverse(
        flmn=flmn,
        L=L,
        N=N,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    if method == "jax":
        f.block_until_ready()
    return f


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
