"""Benchmarks for spherical transforms."""

import numpy as np
import pyssht
import s2fft
from s2fft.recursions.price_mcewen import generate_precomputes
from s2fft.sampling import s2_samples as samples
from benchmarking import benchmark, skip, parse_args_collect_and_run_benchmarks


L_VALUES = [8, 16, 32, 64, 128, 256]
L_LOWER_VALUES = [0]
SPIN_VALUES = [0]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
SPMD_VALUES = [False]


def setup_forward(method, L, L_lower, sampling, spin, reality, spmd):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    f = pyssht.inverse(
        samples.flm_2d_to_1d(flm, L),
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=reality,
    )
    precomps = generate_precomputes(L, spin, sampling, forward=True, L_lower=L_lower)
    return {"f": f, "precomps": precomps}


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
)
def forward(f, precomps, method, L, L_lower, sampling, spin, reality, spmd):
    if method == "pyssht":
        flm = pyssht.forward(f, L, spin, sampling.upper())
    else:
        flm = s2fft.transforms.spherical.forward(
            f=f,
            L=L,
            L_lower=L_lower,
            precomps=precomps,
            spin=spin,
            sampling=sampling,
            reality=reality,
            method=method,
            spmd=spmd,
        )
    if method == "jax":
        flm.block_until_ready()


def setup_inverse(method, L, L_lower, sampling, spin, reality, spmd):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    precomps = generate_precomputes(L, spin, sampling, forward=False, L_lower=L_lower)
    return {"flm": flm, "precomps": precomps}


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
)
def inverse(flm, precomps, method, L, L_lower, sampling, spin, reality, spmd):
    if method == "pyssht":
        f = pyssht.inverse(samples.flm_2d_to_1d(flm, L), L, spin, sampling.upper())
    else:
        f = s2fft.transforms.spherical.inverse(
            flm=flm,
            L=L,
            L_lower=L_lower,
            precomps=precomps,
            spin=spin,
            sampling=sampling,
            reality=reality,
            method=method,
            spmd=spmd,
        )
    if method == "jax":
        f.block_until_ready()


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
