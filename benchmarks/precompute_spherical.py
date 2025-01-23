"""Benchmarks for precompute spherical transforms."""

import numpy as np
import pyssht
from benchmarking import benchmark, parse_args_collect_and_run_benchmarks, skip

import s2fft
import s2fft.precompute_transforms
from s2fft.sampling import s2_samples as samples

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
    f = pyssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=reality,
    )
    kernel_function = (
        s2fft.precompute_transforms.construct.spin_spherical_kernel_jax
        if method == "jax"
        else s2fft.precompute_transforms.construct.spin_spherical_kernel
    )
    kernel = kernel_function(
        L=L,
        spin=spin,
        reality=reality,
        sampling=sampling,
        forward=True,
        recursion=recursion,
    )
    return {"f": f, "kernel": kernel}, flm


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
    flm = s2fft.precompute_transforms.spherical.forward(
        f=f,
        L=L,
        spin=spin,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    if method == "jax":
        flm.block_until_ready()
    return flm


def setup_inverse(method, L, sampling, spin, reality, recursion):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    kernel_function = (
        s2fft.precompute_transforms.construct.spin_spherical_kernel_jax
        if method == "jax"
        else s2fft.precompute_transforms.construct.spin_spherical_kernel
    )
    kernel = kernel_function(
        L=L,
        spin=spin,
        reality=reality,
        sampling=sampling,
        forward=False,
        recursion=recursion,
    )
    return {"flm": flm, "kernel": kernel}, None


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
    f = s2fft.precompute_transforms.spherical.inverse(
        flm=flm,
        L=L,
        spin=spin,
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
