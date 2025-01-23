"""Benchmarks for on-the-fly spherical transforms."""

import jax
import numpy as np
from benchmarking import benchmark, parse_args_collect_and_run_benchmarks, skip

import s2fft
from s2fft.recursions.price_mcewen import generate_precomputes_jax

L_VALUES = [8, 16, 32, 64, 128, 256]
L_LOWER_VALUES = [0]
SPIN_VALUES = [0]
L_TO_NSIDE_RATIO_VALUES = [2]
SAMPLING_VALUES = ["mw"]
METHOD_VALUES = ["numpy", "jax"]
REALITY_VALUES = [True]
SPMD_VALUES = [False]
N_ITER_VALUES = [None]


def _jax_arrays_to_numpy(precomps):
    return [np.asarray(p) for p in precomps]


def _get_nside(sampling, L, L_to_nside_ratio):
    return None if sampling != "healpix" else L // L_to_nside_ratio


def setup_forward(
    method, L, L_lower, sampling, spin, L_to_nside_ratio, reality, spmd, n_iter
):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
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
        spmd=spmd,
        L_lower=L_lower,
    )
    precomps = generate_precomputes_jax(
        L, spin, sampling, nside=nside, forward=True, L_lower=L_lower
    )
    if method == "numpy":
        precomps = _jax_arrays_to_numpy(precomps)
    return {"f": f, "precomps": precomps}, flm


@benchmark(
    setup_forward,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    L_to_nside_ratio=L_TO_NSIDE_RATIO_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
    n_iter=N_ITER_VALUES,
)
def forward(
    f,
    precomps,
    method,
    L,
    L_lower,
    sampling,
    spin,
    L_to_nside_ratio,
    reality,
    spmd,
    n_iter,
):
    flm = s2fft.transforms.spherical.forward(
        f=f,
        L=L,
        L_lower=L_lower,
        precomps=precomps,
        spin=spin,
        nside=_get_nside(sampling, L, L_to_nside_ratio),
        sampling=sampling,
        reality=reality,
        method=method,
        spmd=spmd,
        iter=n_iter,
    )
    return flm.block_until_ready() if isinstance(flm, jax.Array) else flm


def setup_inverse(method, L, L_lower, sampling, spin, L_to_nside_ratio, reality, spmd):
    if reality and spin != 0:
        skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        skip("GPU distribution only valid for JAX.")
    rng = np.random.default_rng()
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, spin=spin, reality=reality)
    precomps = generate_precomputes_jax(
        L,
        spin,
        sampling,
        nside=_get_nside(sampling, L, L_to_nside_ratio),
        forward=False,
        L_lower=L_lower,
    )
    if method == "numpy":
        precomps = _jax_arrays_to_numpy(precomps)
    return {"flm": flm, "precomps": precomps}, None


@benchmark(
    setup_inverse,
    method=METHOD_VALUES,
    L=L_VALUES,
    L_lower=L_LOWER_VALUES,
    sampling=SAMPLING_VALUES,
    spin=SPIN_VALUES,
    L_to_nside_ratio=L_TO_NSIDE_RATIO_VALUES,
    reality=REALITY_VALUES,
    spmd=SPMD_VALUES,
)
def inverse(
    flm, precomps, method, L, L_lower, sampling, spin, L_to_nside_ratio, reality, spmd
):
    f = s2fft.transforms.spherical.inverse(
        flm=flm,
        L=L,
        L_lower=L_lower,
        precomps=precomps,
        spin=spin,
        nside=_get_nside(sampling, L, L_to_nside_ratio),
        sampling=sampling,
        reality=reality,
        method=method,
        spmd=spmd,
    )
    return f.block_until_ready() if isinstance(f, jax.Array) else f


if __name__ == "__main__":
    results = parse_args_collect_and_run_benchmarks()
