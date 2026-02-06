import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from s2fft.recursions.price_mcewen import generate_precomputes_jax
from s2fft.transforms import spherical

jax.config.update("jax_enable_x64", True)

L_to_test = [16]
L_lower_to_test = [2]
spin_to_test = [-2, 0, 1]
nside_to_test = [8]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
reality_to_test = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inverse_custom_gradients(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    reality: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f_target = spherical.inverse_jax(
        flm_target,
        L,
        L_lower=L_lower,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    precomps = generate_precomputes_jax(
        L, spin, sampling, forward=False, L_lower=L_lower
    )

    def func(flm):
        f = spherical.inverse_jax(
            flm,
            L,
            spin=spin,
            L_lower=L_lower,
            reality=reality,
            precomps=precomps,
            sampling=sampling,
        )
        return jnp.sum(jnp.abs(f - f_target) ** 2)

    check_grads(func, (flm,), order=1, modes=("rev"))


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_forward_custom_gradients(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    reality: bool,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f = spherical.inverse_jax(
        flm,
        L,
        L_lower=L_lower,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    precomps = generate_precomputes_jax(
        L, spin, sampling, forward=True, L_lower=L_lower
    )

    def func(f):
        flm = spherical.forward_jax(
            f,
            L,
            spin=spin,
            L_lower=L_lower,
            reality=reality,
            precomps=precomps,
            sampling=sampling,
        )
        return jnp.sum(jnp.abs(flm - flm_target) ** 2)

    check_grads(func, (f,), order=1, modes=("rev"))


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_healpix_inverse_custom_gradients(
    flm_generator,
    nside: int,
    L_lower: int,
    spin: int,
    reality: bool,
):
    sampling = "healpix"
    L = 2 * nside

    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f_target = spherical.inverse_jax(
        flm_target,
        L,
        L_lower=L_lower,
        spin=spin,
        nside=nside,
        sampling=sampling,
        reality=reality,
    )
    precomps = generate_precomputes_jax(
        L, spin, sampling, nside, forward=False, L_lower=L_lower
    )

    def func(flm):
        f = spherical.inverse_jax(
            flm,
            L,
            spin=spin,
            nside=nside,
            L_lower=L_lower,
            reality=reality,
            precomps=precomps,
            sampling=sampling,
        )
        return jnp.sum(jnp.abs(f - f_target) ** 2)

    check_grads(func, (flm,), order=1, modes=("rev"))


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("iter", [0, 1, 2, 3])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_healpix_forward_custom_gradients(
    flm_generator,
    nside: int,
    L_lower: int,
    spin: int,
    reality: bool,
    iter: int,
):
    sampling = "healpix"
    L = 2 * nside

    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f = spherical.inverse_jax(
        flm,
        L,
        L_lower=L_lower,
        spin=spin,
        nside=nside,
        sampling=sampling,
        reality=reality,
    )
    precomps = generate_precomputes_jax(
        L, spin, sampling, nside, forward=True, L_lower=L_lower
    )

    def func(f):
        flm = spherical.forward(
            f,
            L,
            method="jax",
            spin=spin,
            nside=nside,
            L_lower=L_lower,
            reality=reality,
            precomps=precomps,
            sampling=sampling,
            iter=iter,
        )
        return jnp.sum(jnp.abs(flm - flm_target) ** 2)

    check_grads(func, (f,), order=1, modes=("rev"))


@pytest.mark.pyssht
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("_ssht_backend", [0, 1])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ssht_c_backend_inverse_custom_gradients(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    reality: bool,
    _ssht_backend: int,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    if sampling.lower() == "dh" and _ssht_backend == 1:
        pytest.skip("Driscoll Healy ducc0 backend gradient calculation tempremental.")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f_target = spherical.inverse(
        flm_target,
        L,
        spin,
        sampling=sampling,
        method="jax_ssht",
        reality=reality,
        _ssht_backend=_ssht_backend,
    )

    def func(flm):
        f = spherical.inverse(
            flm,
            L,
            spin,
            sampling=sampling,
            method="jax_ssht",
            reality=reality,
            _ssht_backend=_ssht_backend,
        )
        return jnp.sum(jnp.abs(f - f_target) ** 2)

    check_grads(func, (flm,), order=1, modes=("rev"))


@pytest.mark.pyssht
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("L_lower", L_lower_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("_ssht_backend", [0, 1])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ssht_c_backend_forward_custom_gradients(
    flm_generator,
    L: int,
    L_lower: int,
    spin: int,
    sampling: str,
    reality: bool,
    _ssht_backend: int,
):
    if reality and spin != 0:
        pytest.skip("Reality only valid for scalar fields (spin=0).")

    if sampling.lower() == "dh" and _ssht_backend == 1:
        pytest.skip("Driscoll Healy ducc0 backend gradient calculation tempremental.")

    flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    flm_target = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
    f = spherical.inverse(
        flm,
        L,
        spin,
        sampling=sampling,
        method="jax_ssht",
        reality=reality,
        _ssht_backend=_ssht_backend,
    )

    def func(f):
        flm = spherical.forward(
            f,
            L,
            spin,
            sampling=sampling,
            method="jax_ssht",
            reality=reality,
            _ssht_backend=_ssht_backend,
        )
        return jnp.sum(jnp.abs(flm - flm_target) ** 2)

    check_grads(func, (f,), order=1, modes=("rev"))


@pytest.mark.healpy
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_healpix_c_backend_inverse_custom_gradients(flm_generator, nside: int):
    L = 2 * nside
    reality = True
    flm = flm_generator(L=L, reality=reality)

    def func(flm):
        return spherical.inverse(
            flm, L, 0, nside, sampling="healpix", method="jax_healpy", reality=True
        )

    check_grads(func, (flm,), order=2, modes=("fwd", "rev"))


@pytest.mark.healpy
@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("iter", [0, 1, 2, 3])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_healpix_c_backend_forward_custom_gradients(
    flm_generator, nside: int, iter: int
):
    sampling = "healpix"
    L = 2 * nside
    reality = True
    flm = flm_generator(L=L, reality=reality)
    f = spherical.inverse_jax(flm, L, nside=nside, sampling=sampling, reality=reality)

    def func(f):
        return spherical.forward(
            f, L, nside=nside, sampling="healpix", method="jax_healpy", iter=iter
        )

    check_grads(func, (f,), order=2, modes=("fwd", "rev"))
