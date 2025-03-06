"""Benchmarks for precompute Wigner-d transforms."""

import jax
import numpy as np
from benchmarking import (
    BenchmarkSetup,
    benchmark,
    parse_args_collect_and_run_benchmarks,
)

import s2fft
import s2fft.precompute_transforms
from s2fft.base_transforms import wigner as base_wigner
from s2fft.utils import torch_wrapper

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
    # As torch method wraps JAX functions and converting NumPy array to torch tensor
    # causes warning 'DLPack buffer is not aligned' about byte aligment on subsequently
    # converting to JAX array using mutual DLPack support we first convert the NumPy
    # arrays to a JAX arrays before converting to torch tensors which eliminates this
    # warning
    if method.startswith("jax") or method.startswith("torch"):
        flmn = jax.numpy.asarray(flmn)
        f = jax.numpy.asarray(f)
    if method.startswith("torch"):
        flmn, f = torch_wrapper.tree_map_jax_array_to_torch_tensor((flmn, f))
    kernel_function = s2fft.precompute_transforms.wigner._kernel_functions[method]
    kernel = kernel_function(
        L=L, N=N, reality=reality, sampling=sampling, forward=True, mode=mode
    )
    return BenchmarkSetup({"f": f, "kernel": kernel}, flmn, "jax" in method)


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
    return s2fft.precompute_transforms.wigner.forward(
        f=f,
        L=L,
        N=N,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )


def setup_inverse(method, L, N, L_lower, sampling, reality, mode):
    rng = np.random.default_rng()
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L, N, reality=reality)
    # As torch method wraps JAX functions and converting NumPy array to torch tensor
    # causes warning 'DLPack buffer is not aligned' about byte aligment on subsequently
    # converting to JAX array using mutual DLPack support we first convert the NumPy
    # arrays to a JAX arrays before converting to torch tensors which eliminates this
    # warning
    if method.startswith("jax") or method.startswith("torch"):
        flmn = jax.numpy.asarray(flmn)
    if method.startswith("torch"):
        flmn = torch_wrapper.jax_array_to_torch_tensor(flmn)
    kernel_function = s2fft.precompute_transforms.wigner._kernel_functions[method]
    kernel = kernel_function(
        L=L, N=N, reality=reality, sampling=sampling, forward=False, mode=mode
    )
    return BenchmarkSetup({"flmn": flmn, "kernel": kernel}, None, "jax" in method)


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
    return s2fft.precompute_transforms.wigner.inverse(
        flmn=flmn,
        L=L,
        N=N,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
