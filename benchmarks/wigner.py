"""Benchmarks for Wigner transforms."""

import numpy as np
import s2fft
from s2fft.base_transforms import wigner as base_wigner
from s2fft.recursions.price_mcewen import (
    generate_precomputes_wigner,
    generate_precomputes_wigner_jax,
)
from benchmarking import benchmark, skip, parse_args_collect_and_run_benchmarks

L_VALUES = [16, 32, 64, 128, 256]
N_VALUES = [2]
L_LOWER_VALUES = [0]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
SPMD_VALUES = [False]


def setup_forward(method, L, L_lower, N, sampling, reality, spmd):
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
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
    generate_precomputes = (
        generate_precomputes_wigner_jax
        if method == "jax"
        else generate_precomputes_wigner
    )
    precomps = generate_precomputes(
        L, N, sampling, forward=True, reality=reality, L_lower=L_lower
    )
    return {"f": f, "precomps": precomps}


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    N=N_VALUES,
    sampling=SAMPLING_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
)
def forward(f, precomps, method, L, L_lower, N, sampling, reality, spmd):
    flmn = s2fft.transforms.wigner.forward(
        f=f,
        L=L,
        L_lower=L_lower,
        N=N,
        precomps=precomps,
        sampling=sampling,
        reality=reality,
        method=method,
        spmd=spmd,
    )
    if method == "jax":
        flmn.block_until_ready()


def setup_inverse(method, L, L_lower, N, sampling, reality, spmd):
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
    rng = np.random.default_rng()
    flmn = s2fft.utils.signal_generator.generate_flmn(rng, L, N, reality=reality)
    generate_precomputes = (
        generate_precomputes_wigner_jax
        if method == "jax"
        else generate_precomputes_wigner
    )
    precomps = generate_precomputes(
        L, N, sampling, forward=False, reality=reality, L_lower=L_lower
    )
    return {"flmn": flmn, "precomps": precomps}


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    N=N_VALUES,
    sampling=SAMPLING_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
)
def inverse(flmn, precomps, method, L, L_lower, N, sampling, reality, spmd):
    f = s2fft.transforms.spherical.inverse(
        flm=flmn,
        L=L,
        L_lower=L_lower,
        N=N,
        precomps=precomps,
        sampling=sampling,
        reality=reality,
        method=method,
        spmd=spmd,
    )
    if method == "jax":
        f.block_until_ready()


if __name__ == "__main__":
    parse_args_collect_and_run_benchmarks()
