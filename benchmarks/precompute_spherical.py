"""Benchmarks for precompute spherical transforms."""

import jax
import numpy as np
from benchmarking import (
    BenchmarkSetup,
    benchmark,
    parse_args_collect_and_run_benchmarks,
    skip,
)

import s2fft
import s2fft.precompute_transforms
from s2fft.utils import torch_wrapper

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
    # As torch method wraps JAX functions and converting NumPy array to torch tensor
    # causes warning 'DLPack buffer is not aligned' about byte aligment on subsequently
    # converting to JAX array using mutual DLPack support we first convert the NumPy
    # arrays to a JAX arrays before converting to torch tensors which eliminates this
    # warning
    if method.startswith("jax") or method.startswith("torch"):
        flm = jax.numpy.asarray(flm)
        f = jax.numpy.asarray(f)
    if method.startswith("torch"):
        flm, f = torch_wrapper.tree_map_jax_array_to_torch_tensor((flm, f))
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
    # As torch method wraps JAX functions and converting NumPy array to torch tensor
    # causes warning 'DLPack buffer is not aligned' about byte aligment on subsequently
    # converting to JAX array using mutual DLPack support we first convert the NumPy
    # array to a JAX array before converting to a torch tensor which eliminates this
    # warning
    if method.startswith("jax") or method.startswith("torch"):
        flm = jax.numpy.asarray(flm)
    if method.startswith("torch"):
        flm = torch_wrapper.jax_array_to_torch_tensor(flm)
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
