"""Tests for the custom JAX primitives backing the latitudinal step.

These tests verify:

1. **Numerical equivalence** — the primitive matches a direct call to the
   underlying ``otf`` function for both directions and several samplings.
2. **Reverse-mode AD (grad)** — ``check_grads`` confirms the transpose rule
   produces correct cotangents.
3. **vmap** — the batcher correctly maps over a leading batch dim and the
   result equals an explicit Python loop.
4. **Composed transforms** — ``vmap(grad(...))`` and ``grad(vmap(...))``
   both produce the same result as a manual loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from s2fft.recursions.price_mcewen import generate_precomputes_jax
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import _ftm_flm_primitive as ftm_flm_prim
from s2fft.transforms import otf_recursions as otf

jax.config.update("jax_enable_x64", True)


# Light parameter sweep — primitive code paths don't depend on L numerically,
# so a single small L plus a couple of samplings gives broad coverage cheaply.
L_TEST = 8
L_LOWER_TEST = 0
SPIN_TEST = [-2, 0, 1]
SAMPLING_TEST = ["mw", "mwss", "dh", "gl"]


def _make_inputs(rng, L, spin, sampling, reality):
    flm = (
        rng.standard_normal(samples.flm_shape(L))
        + 1j * rng.standard_normal(samples.flm_shape(L))
    ).astype(np.complex128)
    ftm = (
        rng.standard_normal(samples.ftm_shape(L, sampling))
        + 1j * rng.standard_normal(samples.ftm_shape(L, sampling))
    ).astype(np.complex128)
    if reality:
        flm = flm.real.astype(np.complex128)
        ftm = ftm.real.astype(np.complex128)
    thetas = jnp.asarray(samples.thetas(L, sampling))
    return jnp.asarray(flm), jnp.asarray(ftm), thetas


# ---------------------------------------------------------------------------
# 1. Numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spin", SPIN_TEST)
@pytest.mark.parametrize("sampling", SAMPLING_TEST)
def test_flm_to_ftm_matches_otf(spin, sampling, rng):
    flm, _, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)
    expected = otf.inverse_latitudinal_step_jax(
        flm,
        thetas,
        L_TEST,
        spin,
        None,
        sampling,
        False,
        precomps=None,
        spmd=False,
        L_lower=L_LOWER_TEST,
    )
    actual = ftm_flm_prim.flm_to_ftm(
        flm,
        thetas,
        L=L_TEST,
        spin=spin,
        nside=None,
        sampling=sampling,
        reality=False,
        spmd=False,
        L_lower=L_LOWER_TEST,
        precomps=None,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


@pytest.mark.parametrize("spin", SPIN_TEST)
@pytest.mark.parametrize("sampling", SAMPLING_TEST)
def test_ftm_to_flm_matches_otf(spin, sampling, rng):
    _, ftm, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)
    expected = otf.forward_latitudinal_step_jax(
        ftm,
        thetas,
        L_TEST,
        spin,
        None,
        sampling,
        False,
        precomps=None,
        spmd=False,
        L_lower=L_LOWER_TEST,
    )
    actual = ftm_flm_prim.ftm_to_flm(
        ftm,
        thetas,
        L=L_TEST,
        spin=spin,
        nside=None,
        sampling=sampling,
        reality=False,
        spmd=False,
        L_lower=L_LOWER_TEST,
        precomps=None,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


@pytest.mark.parametrize("sampling", SAMPLING_TEST)
def test_flm_to_ftm_matches_with_explicit_precomps(sampling, rng):
    spin = 1
    flm, _, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)
    precomps = generate_precomputes_jax(
        L_TEST,
        spin,
        sampling,
        None,
        forward=False,
        L_lower=L_LOWER_TEST,
        betas=thetas,
    )
    expected = otf.inverse_latitudinal_step_jax(
        flm,
        thetas,
        L_TEST,
        spin,
        None,
        sampling,
        False,
        precomps=precomps,
        spmd=False,
        L_lower=L_LOWER_TEST,
    )
    actual = ftm_flm_prim.flm_to_ftm(
        flm,
        thetas,
        L=L_TEST,
        spin=spin,
        nside=None,
        sampling=sampling,
        reality=False,
        spmd=False,
        L_lower=L_LOWER_TEST,
        precomps=precomps,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Reverse-mode AD via the transpose rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spin", SPIN_TEST)
@pytest.mark.parametrize("sampling", SAMPLING_TEST)
def test_flm_to_ftm_grad(spin, sampling, rng):
    flm, _, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)

    def flm_to_ftm(flm):
        return ftm_flm_prim.flm_to_ftm(
            flm,
            thetas,
            L=L_TEST,
            spin=spin,
            nside=None,
            sampling=sampling,
            reality=False,
            spmd=False,
            L_lower=L_LOWER_TEST,
            precomps=None,
        )

    check_grads(flm_to_ftm, (flm,), order=2, modes=("fwd", "rev"))


@pytest.mark.parametrize("spin", SPIN_TEST)
@pytest.mark.parametrize("sampling", SAMPLING_TEST)
def test_ftm_to_flm_grad(spin, sampling, rng):
    _, ftm, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)

    def ftm_to_flm(ftm):
        return ftm_flm_prim.ftm_to_flm(
            ftm,
            thetas,
            L=L_TEST,
            spin=spin,
            nside=None,
            sampling=sampling,
            reality=False,
            spmd=False,
            L_lower=L_LOWER_TEST,
            precomps=None,
        )

    # TODO: figure out why this fails for MWSS scheme for order=2
    check_grads(ftm_to_flm, (ftm,), order=1, modes=("fwd", "rev"))


def test_grad_matches_direct_otf_call(rng):
    """The gradient via our primitive must match the gradient via a manual
    custom_vjp around the same underlying functions, which is the formula
    used by the original code."""
    spin = 1
    sampling = "mw"
    flm, _, thetas = _make_inputs(rng, L_TEST, spin, sampling, reality=False)
    glm = jnp.asarray(
        np.random.default_rng(7).standard_normal(samples.ftm_shape(L_TEST, sampling))
        + 1j
        * np.random.default_rng(8).standard_normal(samples.ftm_shape(L_TEST, sampling))
    )

    def via_primitive(flm):
        ftm = ftm_flm_prim.flm_to_ftm(
            flm,
            thetas,
            L=L_TEST,
            spin=spin,
            nside=None,
            sampling=sampling,
            reality=False,
            spmd=False,
            L_lower=L_LOWER_TEST,
            precomps=None,
        )
        return jnp.real(jnp.vdot(glm, ftm))

    grad_prim = jax.grad(via_primitive)(flm)
    # The expected gradient: jax.vjp of inverse_latitudinal_step_jax pulled
    # back via the analytic transpose (forward_latitudinal_step_jax).
    expected = otf.forward_latitudinal_step_jax(
        glm,
        thetas,
        L_TEST,
        spin,
        None,
        sampling,
        False,
        precomps=None,
        spmd=False,
        L_lower=L_LOWER_TEST,
    )
    # ``jax.grad`` of a real-valued function of complex arrays returns the
    # conjugate of the holomorphic gradient. Compare against the conjugate.
    np.testing.assert_allclose(
        np.asarray(grad_prim),
        np.asarray(jnp.conj(expected)),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 3. vmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["mw", "healpix"])
def test_flm_to_ftm_vmap_matches_loop(sampling, rng):
    spin = 0
    nside = 4 if sampling == "healpix" else None
    L = 2 * nside if sampling == "healpix" else L_TEST
    batch = 3
    flms = (
        rng.standard_normal((batch,) + samples.flm_shape(L))
        + 1j * rng.standard_normal((batch,) + samples.flm_shape(L))
    ).astype(np.complex128)
    flms = jnp.asarray(flms)
    thetas = jnp.asarray(samples.thetas(L, sampling, nside))

    def call(flm):
        return ftm_flm_prim.flm_to_ftm(
            flm,
            thetas,
            L=L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            reality=False,
            spmd=False,
            L_lower=0,
            precomps=None,
        )

    vmapped = jax.vmap(call)(flms)
    looped = jnp.stack([call(flms[i]) for i in range(batch)])
    np.testing.assert_allclose(np.asarray(vmapped), np.asarray(looped), atol=1e-12)


@pytest.mark.parametrize("sampling", ["mw", "healpix"])
def test_ftm_to_flm_vmap_matches_loop(sampling, rng):
    spin = 0
    nside = 4 if sampling == "healpix" else None
    L = 2 * nside if sampling == "healpix" else L_TEST
    batch = 3
    ftms = (
        rng.standard_normal((batch,) + samples.ftm_shape(L, sampling, nside))
        + 1j * rng.standard_normal((batch,) + samples.ftm_shape(L, sampling, nside))
    ).astype(np.complex128)
    ftms = jnp.asarray(ftms)
    thetas = jnp.asarray(samples.thetas(L, sampling, nside))

    def call(ftm):
        return ftm_flm_prim.ftm_to_flm(
            ftm,
            thetas,
            L=L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            reality=False,
            spmd=False,
            L_lower=0,
            precomps=None,
        )

    vmapped = jax.vmap(call)(ftms)
    looped = jnp.stack([call(ftms[i]) for i in range(batch)])
    np.testing.assert_allclose(np.asarray(vmapped), np.asarray(looped), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. vmap composed with grad (the case that broke under custom_vjp / linear_call)
# ---------------------------------------------------------------------------


def test_grad_of_vmap_flm_to_ftm(rng):
    spin = 0
    sampling = "mw"
    L = L_TEST
    batch = 3
    flms = jnp.asarray(
        rng.standard_normal((batch,) + samples.flm_shape(L))
        + 1j * rng.standard_normal((batch,) + samples.flm_shape(L))
    )
    thetas = jnp.asarray(samples.thetas(L, sampling))

    def loss(flms):
        ftms = jax.vmap(
            lambda flm: ftm_flm_prim.flm_to_ftm(
                flm,
                thetas,
                L=L,
                spin=spin,
                nside=None,
                sampling=sampling,
                reality=False,
                spmd=False,
                L_lower=0,
                precomps=None,
            )
        )(flms)
        return jnp.sum(jnp.abs(ftms) ** 2)

    # Should not raise; gradient should match a Python-loop reference.
    grad_via_vmap = jax.grad(loss)(flms)

    def loss_loop(flms):
        total = 0.0
        for i in range(batch):
            ftm = ftm_flm_prim.flm_to_ftm(
                flms[i],
                thetas,
                L=L,
                spin=spin,
                nside=None,
                sampling=sampling,
                reality=False,
                spmd=False,
                L_lower=0,
                precomps=None,
            )
            total = total + jnp.sum(jnp.abs(ftm) ** 2)
        return total

    grad_via_loop = jax.grad(loss_loop)(flms)
    np.testing.assert_allclose(
        np.asarray(grad_via_vmap),
        np.asarray(grad_via_loop),
        atol=1e-10,
    )


def test_vmap_of_grad_ftm_to_flm(rng):
    spin = 0
    sampling = "mw"
    L = L_TEST
    batch = 3
    ftms = jnp.asarray(
        rng.standard_normal((batch,) + samples.ftm_shape(L, sampling))
        + 1j * rng.standard_normal((batch,) + samples.ftm_shape(L, sampling))
    )
    thetas = jnp.asarray(samples.thetas(L, sampling))

    def per_sample_grad(ftm):
        return jax.grad(
            lambda f: jnp.sum(
                jnp.abs(
                    ftm_flm_prim.ftm_to_flm(
                        f,
                        thetas,
                        L=L,
                        spin=spin,
                        nside=None,
                        sampling=sampling,
                        reality=False,
                        spmd=False,
                        L_lower=0,
                        precomps=None,
                    )
                )
                ** 2
            )
        )(ftm)

    via_vmap = jax.vmap(per_sample_grad)(ftms)
    via_loop = jnp.stack([per_sample_grad(ftms[i]) for i in range(batch)])
    np.testing.assert_allclose(np.asarray(via_vmap), np.asarray(via_loop), atol=1e-10)
