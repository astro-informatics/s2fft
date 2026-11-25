from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from s2fft import recursions
from s2fft.base_transforms import spherical as base
from s2fft.precompute_transforms import spherical as precompute_spherical
from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
from s2fft.recursions.price_mcewen import generate_precomputes
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical

jax.config.update("jax_enable_x64", True)

L_to_test = [6, 7]
L_lower_to_test = [0, 2]
spin_to_test = [-2, 0, 1]
nside_to_test = [4, 5]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
method_to_test = ["numpy", "jax", "torch"]
reality_to_test = [False, True]
multiple_gpus = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("use_generate_precomputes", [True, False])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse(
    cached_ssht_test_case: Callable,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
    use_generate_precomputes: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    test_data = cached_ssht_test_case(L, L_lower, spin, sampling, reality)

    if use_generate_precomputes:
        precomps = generate_precomputes(L, spin, sampling, L_lower=L_lower)
    else:
        precomps = None
    f = spherical.inverse(
        torch.from_numpy(test_data["flm"]) if method == "torch" else test_data["flm"],
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(f, test_data["f_ssht"], atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse_healpix(
    cached_healpy_test_case: Callable,
    nside: int,
    method: str,
    spmd: bool,
):
    sampling = "healpix"
    L = 2 * nside
    reality = True
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    precomps = generate_precomputes(L, 0, sampling, nside, False)
    f = spherical.inverse(
        torch.from_numpy(test_data["flm"]) if method == "torch" else test_data["flm"],
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
    )

    np.testing.assert_allclose(np.real(f), np.real(test_data["f_hp"]), atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_inverse_healpix_spin(
    flm_generator: Callable,
    nside: int,
    spin: int,
    method: str,
):
    # Regression test for the HEALPix spin != 0 inverse transform. The recursive
    # (on-the-fly) Wigner-d recursion previously hit exact zero nodes at the
    # rational cos(theta) values of HEALPix rings, injecting NaNs that nansum
    # dropped, producing percent-level errors. Validate against the independent
    # Turok-recursion base transform which samples the same HEALPix grid.
    sampling = "healpix"
    L = 2 * nside
    flm = flm_generator(L=L, spin=spin, reality=False)
    f = spherical.inverse(
        flm, L, spin=spin, nside=nside, sampling=sampling, method=method
    )
    f_base = base.inverse(flm, L, spin=spin, nside=nside, sampling=sampling)
    np.testing.assert_allclose(np.asarray(f), f_base, atol=1e-12)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("use_generate_precomputes", [True, False])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward(
    cached_ssht_test_case: Callable,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    method: str,
    reality: bool,
    spmd: bool,
    use_generate_precomputes: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")
    if spmd and method != "jax":
        pytest.skip("GPU distribution only valid for JAX.")

    test_data = cached_ssht_test_case(L, L_lower, spin, sampling, reality)

    if use_generate_precomputes:
        precomps = generate_precomputes(L, spin, sampling, None, True, L_lower)
    else:
        precomps = None

    flm_check = spherical.forward(
        torch.from_numpy(test_data["f_ssht"])
        if method == "torch"
        else test_data["f_ssht"],
        L,
        spin,
        sampling=sampling,
        method=method,
        reality=reality,
        precomps=precomps,
        spmd=spmd,
        L_lower=L_lower,
    )
    np.testing.assert_allclose(test_data["flm"], flm_check, atol=1e-14)


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("method", method_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
@pytest.mark.parametrize("iter", [0, 1, 2, 3])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_forward_healpix(
    cached_healpy_test_case: Callable,
    nside: int,
    method: str,
    spmd: bool,
    iter: int,
):
    sampling = "healpix"
    L = 2 * nside
    reality = True
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality, n_iter=iter)

    precomps = generate_precomputes(L, 0, sampling, nside, True)
    flm_check = spherical.forward(
        torch.from_numpy(test_data["f_hp"]) if method == "torch" else test_data["f_hp"],
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
        iter=iter,
    )
    flm_check = samples.flm_2d_to_hp(flm_check, L)

    np.testing.assert_allclose(test_data["flm_hp"], flm_check, atol=1e-14)


@pytest.mark.parametrize("spin", [8, 12, -12])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl", "healpix"])
@pytest.mark.parametrize("method", ["numpy", "jax"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_transform_high_spin(
    flm_generator: Callable, spin: int, sampling: str, method: str
):
    # Regression test for issue #176: the on-the-fly recursion lost accuracy on the
    # rings nearest the poles beyond |spin| ~ 4, so that spin >= 8 was refused. Validate
    # against the precompute transforms with kernels from the independent Risbo
    # recursion.
    L = 24
    nside = L // 2 if sampling == "healpix" else None
    flm = flm_generator(L=L, spin=spin, reality=False)

    kernel = spin_spherical_kernel_jax(L, spin, False, sampling, nside, False, "risbo")
    f_check = precompute_spherical.inverse(
        flm, L, spin, kernel, sampling, False, "jax", nside
    )
    f = spherical.inverse(
        flm, L, spin=spin, nside=nside, sampling=sampling, method=method
    )
    np.testing.assert_allclose(f, f_check, rtol=0, atol=1e-12)

    kernel = spin_spherical_kernel_jax(L, spin, False, sampling, nside, True, "risbo")
    flm_check = precompute_spherical.forward(
        f_check, L, spin, kernel, sampling, False, "jax", nside
    )
    flm_recov = spherical.forward(
        f_check, L, spin=spin, nside=nside, sampling=sampling, method=method
    )
    np.testing.assert_allclose(flm_recov, flm_check, rtol=0, atol=1e-12)


@pytest.mark.parametrize("spin", [-2, 1, 2])
@pytest.mark.parametrize("method", ["numpy", "jax"])
def test_forward_reality_matches_complex(
    rng: np.random.Generator, spin: int, method: str
):
    # For a real map on GL sampling (no periodic extension in theta), the m >= 0
    # coefficients are the same linear function of the map with reality=True and False
    L = 8
    f = rng.standard_normal((L, 2 * L - 1))
    flm = spherical.forward(
        f, L, spin=spin, sampling="gl", method=method, reality=False
    )
    flm_real = spherical.forward(
        f, L, spin=spin, sampling="gl", method=method, reality=True
    )
    np.testing.assert_allclose(
        flm_real[:, L - 1 :], flm[:, L - 1 :], rtol=0, atol=1e-13
    )


def _risbo_planes(betas: np.ndarray, L: int) -> np.ndarray:
    """Wigner-d planes d^el_{m, m'}(beta) by Risbo, indexed [t, el, L - 1 + m, L - 1 + m']."""
    planes = np.zeros((len(betas), L, 2 * L - 1, 2 * L - 1))
    for t, beta in enumerate(betas):
        dl = np.zeros((2 * L - 1, 2 * L - 1))
        for el in range(L):
            dl = recursions.risbo.compute_full(dl, beta, L, el)
            planes[t, el] = dl
    return planes


def _real_spin_signal(
    rng: np.random.Generator, L: int, spin: int, sampling: str
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    A real signal on the grid of `sampling` and its spin-`spin` coefficients.

    The signal is :math:`f = \sin^{|s|}\theta\, h` with :math:`s` = `spin` and :math:`h`
    a real spin-0 signal of band-limit :math:`L - |s|`. Its :math:`m`-th Fourier row is
    :math:`\sin^{|m| + |s|}\theta` times a polynomial in :math:`\cos\theta` of degree at
    most :math:`L - 1 - |m| - |s|`, which lies in the span of :math:`d^\ell_{m, -s}` and
    in that of :math:`d^\ell_{m, s}` for :math:`\ell < L`, so :math:`f` is band-limited
    at :math:`L` for spin :math:`s` and for spin :math:`-s`.

    The signal and its coefficients are both evaluated by direct sums over Risbo's
    Wigner-d, without the transforms: the coefficients by Gauss-Legendre quadrature with
    :math:`L` nodes and :math:`2L - 1` points in :math:`\phi`, which is exact for
    :math:`f` times a spin-:math:`s` harmonic of degree :math:`\ell < L`, with
    :math:`{}_sY_{\ell m} = (-1)^s \sqrt{(2\ell + 1)/(4\pi)}\, e^{im\phi}\,
    d^\ell_{m, -s}(\theta)`.
    """
    hlm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for el in range(L - abs(spin)):
        hlm[el, L - 1] = rng.standard_normal()
        for m in range(1, el + 1):
            hlm[el, L - 1 + m] = rng.standard_normal() + 1j * rng.standard_normal()
            hlm[el, L - 1 - m] = (-1) ** m * np.conj(hlm[el, L - 1 + m])
    norm = np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    m = np.arange(-(L - 1), L)

    def signal(thetas: np.ndarray, phis: np.ndarray) -> np.ndarray:
        d = _risbo_planes(thetas, L)[..., L - 1]
        h = np.einsum("lm,l,tlm,pm->tp", hlm, norm, d, np.exp(1j * np.outer(phis, m)))
        return np.sin(thetas)[:, None] ** abs(spin) * h.real

    x, weights = np.polynomial.legendre.leggauss(L)
    thetas = np.arccos(x)
    phis = 2 * np.pi * np.arange(2 * L - 1) / (2 * L - 1)
    ftm = signal(thetas, phis) @ np.exp(-1j * np.outer(phis, m)) * 2 * np.pi / len(phis)
    d = _risbo_planes(thetas, L)[..., L - 1 - spin]
    flm = (-1) ** spin * norm[:, None] * np.einsum("t,tlm,tm->lm", weights, d, ftm)
    flm[: abs(spin)] = 0
    flm[np.abs(m)[None, :] > np.arange(L)[:, None]] = 0
    f = signal(samples.thetas(L, sampling), samples.phis_equiang(L, sampling))
    return f, flm


@pytest.mark.parametrize("spin", [-3, -2, -1, 1, 2, 3])
@pytest.mark.parametrize("sampling", ["gl", "mw"])
@pytest.mark.parametrize("method", ["numpy", "jax"])
def test_forward_reality_spin_all_columns(
    rng: np.random.Generator, spin: int, sampling: str, method: str
):
    # For a real signal with spin s != 0, the m < 0 coefficients are not conjugates of
    # the m > 0 ones but f^s_{el, -m} = (-1)^(s + m) conj(f^{-s}_{el, m}), so reality=True
    # must give reality=False's coefficients on all 2L - 1 columns. reality=False is
    # checked against coefficients computed without the transforms.
    L = 8
    f, flm_check = _real_spin_signal(rng, L, spin, sampling)
    flm = spherical.forward(
        f, L, spin=spin, sampling=sampling, method=method, reality=False
    )
    flm_real = spherical.forward(
        f, L, spin=spin, sampling=sampling, method=method, reality=True
    )
    np.testing.assert_allclose(flm, flm_check, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flm_real, flm, rtol=0, atol=1e-12)


@pytest.mark.parametrize("spin", [-3, -2, -1, 1, 2, 3])
@pytest.mark.parametrize("sampling", ["gl", "mw"])
@pytest.mark.parametrize("method", ["numpy", "jax"])
def test_inverse_reality_spin(
    rng: np.random.Generator, spin: int, sampling: str, method: str
):
    # The inverse with reality=True must give the real signal whose spin s != 0
    # coefficients (all 2L - 1 columns) it is passed, as reality=False does.
    # reality=False is checked against the signal computed without the transforms.
    L = 8
    f_check, flm = _real_spin_signal(rng, L, spin, sampling)
    f = spherical.inverse(
        flm, L, spin=spin, sampling=sampling, method=method, reality=False
    )
    f_real = spherical.inverse(
        flm, L, spin=spin, sampling=sampling, method=method, reality=True
    )
    np.testing.assert_allclose(f, f_check, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f_real, f, rtol=0, atol=1e-12)


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl"])
def test_reality_vmap_over_mixed_spins(rng: np.random.Generator, sampling: str):
    # With reality=True the JAX transforms skip half recursion 0 at spin 0 only, by a
    # lax.cond on the spin, which they trace. Under vmap over spins that mix zero and
    # non-zero values the cond becomes a select, so each batched result must equal its
    # own unbatched call, and reality=False's coefficients and signal on all columns.
    L = 8
    spins = np.array([0, 1, -2, 0, 3])
    signals = [_real_spin_signal(rng, L, int(spin), sampling) for spin in spins]
    fs = np.stack([f for f, _ in signals])
    flms = np.stack([flm for _, flm in signals])

    def forward(f, spin, reality=True):
        return spherical.forward_jax(f, L, spin, None, sampling, reality)

    def inverse(flm, spin, reality=True):
        return spherical.inverse_jax(flm, L, spin, None, sampling, reality)

    flm_batched = jax.vmap(forward)(fs, spins)
    f_batched = jax.vmap(inverse)(flms, spins)
    for k, spin in enumerate(spins.tolist()):
        np.testing.assert_allclose(
            flm_batched[k], forward(fs[k], spin), rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            f_batched[k], inverse(flms[k], spin), rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            flm_batched[k], forward(fs[k], spin, False), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            f_batched[k], inverse(flms[k], spin, False), rtol=0, atol=1e-12
        )

    # The transforms are linear, so a jvp through the batched transforms gives the
    # batched transforms of the tangents.
    _, flm_tangent = jax.jvp(
        lambda f: jax.vmap(forward)(f, spins), (fs,), (fs[::-1].copy(),)
    )
    np.testing.assert_allclose(
        flm_tangent, jax.vmap(forward)(fs[::-1].copy(), spins), rtol=0, atol=1e-14
    )
    _, f_tangent = jax.jvp(
        lambda flm: jax.vmap(inverse)(flm, spins), (flms,), (flms[::-1].copy(),)
    )
    np.testing.assert_allclose(
        f_tangent, jax.vmap(inverse)(flms[::-1].copy(), spins), rtol=0, atol=1e-14
    )


def test_reality_vmap_over_signals_keeps_the_spin_zero_skip():
    # Under vmap over the signals alone the spin stays unbatched, so at spin 0 with
    # reality=True half recursion 0 is skipped by a lax.cond, lowered as a case, and
    # not run for every element; a batched spin turns it into a select.
    L = 8
    fs = jnp.zeros((3, L, 2 * L - 1))
    flms = jnp.zeros((3, L, 2 * L - 1), dtype=jnp.complex128)
    spins = jnp.zeros(3, dtype=int)

    def lowered(fun, *args):
        return jax.jit(jax.vmap(fun)).lower(*args).as_text()

    def forward(f, spin=0):
        return spherical.forward_jax(f, L, spin, None, "gl", True)

    def inverse(flm, spin=0):
        return spherical.inverse_jax(flm, L, spin, None, "gl", True)

    assert "stablehlo.case" in lowered(forward, fs)
    assert "stablehlo.case" in lowered(inverse, flms)
    assert "stablehlo.case" not in lowered(forward, fs, spins)
    assert "stablehlo.case" not in lowered(inverse, flms, spins)


@pytest.mark.healpy
@pytest.mark.pyssht
def test_sampling_ssht_backend_exceptions(flm_generator):
    sampling = "healpix"
    nside = 6
    L = 2 * nside

    flm = flm_generator(L=L, reality=False)
    f = spherical.inverse(flm, L, 0, nside, sampling, "jax_healpy")

    with pytest.raises(ValueError):
        spherical.inverse(flm, L, 0, nside, sampling, "jax_ssht")

    with pytest.raises(ValueError):
        spherical.forward(f, L, 0, nside, sampling, "jax_ssht")


@pytest.mark.healpy
@pytest.mark.pyssht
def test_sampling_healpy_backend_exceptions(flm_generator):
    sampling = "mw"
    L = 12

    flm = flm_generator(L=L, reality=False)
    f = spherical.inverse(flm, L, 0, None, sampling, "jax_ssht")

    with pytest.raises(ValueError):
        spherical.inverse(flm, L, 0, None, sampling, "jax_healpy")

    with pytest.raises(ValueError):
        spherical.forward(f, L, 0, None, sampling, "jax_healpy")


def test_sampling_exceptions(flm_generator):
    with pytest.raises(ValueError):
        spherical.inverse(None, 0, 0, None, method="incorrect")

    with pytest.raises(ValueError):
        spherical.forward(None, 0, 0, None, method="incorrect")
