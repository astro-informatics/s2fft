import os
import subprocess
import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from s2fft import recursions
from s2fft.sampling import s2_samples as samples

jax.config.update("jax_enable_x64", True)

L_to_test = [6, 7]
spin_to_test = [-2, 0, 1]
sampling_schemes = ["mw", "mwss", "dh", "gl", "healpix"]


@pytest.fixture
def cached_pyssht_generate_dl(cached_test_case_wrapper: Callable) -> Callable:
    def generate_data(beta: float, L: int) -> dict[str, np.ndarray]:
        import pyssht

        return {"dl": pyssht.generate_dl(beta, L)}

    return cached_test_case_wrapper(generate_data, format="npz")


def test_trapani_with_ssht(cached_pyssht_generate_dl: Callable):
    """Test Trapani computation against ssht"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compute using SSHT.
    beta = np.pi / 2.0
    ssht_data = cached_pyssht_generate_dl(beta, L)

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = recursions.trapani.init(dl, L)
    for el in range(1, L):
        dl = recursions.trapani.compute_full_loop(dl, L, el)
        np.testing.assert_allclose(ssht_data["dl"][el, :, :], dl, atol=1e-10)


def test_trapani_vectorized():
    """Test vectorized Trapani computation"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = recursions.trapani.init(dl, L)
    dl_vect = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl_vect = recursions.trapani.init(dl_vect, L)
    for el in range(1, L):
        dl = recursions.trapani.compute_full_loop(dl, L, el)
        dl_vect = recursions.trapani.compute_full_vectorized(dl_vect, L, el)
        np.testing.assert_allclose(
            dl[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            dl_vect[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            atol=1e-10,
        )


def test_trapani_jax():
    """Test JAX Trapani computation"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = recursions.trapani.init(dl, L)
    dl_jax = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl_jax = recursions.trapani.init_jax(dl_jax, L)
    for el in range(1, L):
        dl = recursions.trapani.compute_full_vectorized(dl, L, el)
        dl_jax = recursions.trapani.compute_full_jax(dl_jax, L, el)
        np.testing.assert_allclose(
            dl[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            dl_jax[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            atol=1e-10,
        )


def test_trapani_interfaces():
    """Test Trapani interfaces"""

    # Test all dl(pi/2) terms up to L.
    L = 5

    dl_loop = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl_loop = recursions.trapani.init(dl_loop, L, implementation="loop")

    dl_vect = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl_vect = recursions.trapani.init(dl_vect, L, implementation="vectorized")

    dl_jax = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl_jax = recursions.trapani.init(dl_jax, L, implementation="jax")

    for el in range(1, L):
        dl_loop = recursions.trapani.compute_full(dl_loop, L, el, implementation="loop")
        dl_vect = recursions.trapani.compute_full(
            dl_vect, L, el, implementation="vectorized"
        )
        dl_jax = recursions.trapani.compute_full(dl_jax, L, el, implementation="jax")
        np.testing.assert_allclose(
            dl_loop[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            dl_vect[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            dl_vect[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            dl_jax[
                -el + (L - 1) : el + (L - 1) + 1,
                -el + (L - 1) : el + (L - 1) + 1,
            ],
            atol=1e-10,
        )

    with pytest.raises(ValueError) as _:
        recursions.trapani.init(dl_loop, L, implementation="unexpected")

    with pytest.raises(ValueError) as _:
        recursions.trapani.compute_full(dl_jax, L, el, implementation="unexpected")


def test_risbo_with_ssht(cached_pyssht_generate_dl: Callable):
    """Test Risbo computation against ssht"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compute using SSHT.
    # beta = np.pi / 2.0
    beta = np.pi / L
    ssht_data = cached_pyssht_generate_dl(beta, L)

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

    for el in range(0, L):
        dl = recursions.risbo.compute_full(dl, beta, L, el)
        np.testing.assert_allclose(ssht_data["dl"][el, :, :], dl, atol=1e-15)


def test_risbo_with_ssht_jax(cached_pyssht_generate_dl: Callable):
    """Test Risbo JAX computation against ssht"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compute using SSHT.
    # betas = [0, np.pi / 2.0, np.pi]
    betas = [np.pi / L]
    for beta in betas:
        ssht_data = cached_pyssht_generate_dl(beta, L)

        # Compare to routines in SSHT, which have been validated extensively.
        dl = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)

        for el in range(0, L):
            dl = recursions.risbo_jax.compute_full(dl, beta, L, el)
            np.testing.assert_allclose(ssht_data["dl"][el, :, :], dl, atol=1e-15)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "healpix"])
def test_turok_with_ssht(cached_pyssht_generate_dl: Callable, L: int, sampling: str):
    """Test Turok computation against ssht"""

    # Test all dl() terms up to L.
    betas = samples.thetas(L, sampling, int(L / 2))

    # Compute using SSHT.
    for beta in betas:
        ssht_data = cached_pyssht_generate_dl(beta, L)

        for el in range(L):
            dl_turok = recursions.turok.compute_full(beta, el, L)

            np.testing.assert_allclose(dl_turok, ssht_data["dl"][el], atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_turok_slice_with_ssht(
    cached_pyssht_generate_dl: Callable, L: int, spin: int, sampling: str
):
    """Test Turok spin slice computation against ssht"""

    # Test all dl() terms up to L.
    betas = samples.thetas(L, sampling, int(L / 2))

    # Compute using SSHT.
    for beta in betas:
        ssht_data = cached_pyssht_generate_dl(beta, L)

        for el in range(L):
            if el >= np.abs(spin):
                dl_turok = recursions.turok.compute_slice(beta, el, L, -spin)

                np.testing.assert_allclose(
                    dl_turok,
                    ssht_data["dl"][el, :, L - 1 - spin],
                    atol=1e-10,
                    rtol=1e-12,
                )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_turok_slice_jax_with_ssht(
    cached_pyssht_generate_dl: Callable, L: int, spin: int, sampling: str
):
    """Test Turok spin slice computation against ssht"""

    # Test all dl() terms up to L.
    betas = samples.thetas(L, sampling, int(L / 2))

    # Compute using SSHT.
    for beta in betas:
        ssht_data = cached_pyssht_generate_dl(beta, L)

        for el in range(L):
            if el >= np.abs(spin):
                print(f"beta {beta}, el {el}, spin {spin}")
                dl_turok = recursions.turok_jax.compute_slice(beta, el, L, -spin)

                np.testing.assert_allclose(
                    dl_turok[L - 1 - el : L - 1 + el + 1],
                    ssht_data["dl"][el, L - 1 - el : L - 1 + el + 1, L - 1 - spin],
                    atol=1e-10,
                    rtol=1e-12,
                )


def test_turok_exceptions():
    L = 10

    with pytest.raises(ValueError) as _:
        recursions.turok.compute_full(np.pi / 2, L, L)

    with pytest.raises(ValueError) as _:
        recursions.turok.compute_slice(beta=np.pi / 2, el=L - 1, L=L, mm=L)

    with pytest.raises(ValueError) as _:
        recursions.turok.compute_slice(beta=np.pi / 2, el=L, L=L, mm=0)


def _risbo_slice(beta: float, L: int, mm: int) -> np.ndarray:
    """Slice d^el_{m, mm}(beta) for all el < L, indexed [L - 1 + m, el], by Risbo."""
    dl = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl_slice = np.zeros((2 * L - 1, L))
    for el in range(L):
        dl = recursions.risbo_jax.compute_full(dl, beta, L, el)
        dl_slice[:, el] = dl[:, L - 1 + mm]
    return dl_slice


def _thetas_nearest_poles(L: int, sampling: str, nside: int = None) -> np.ndarray:
    """Two sample rings nearest each pole, excluding rings on the poles themselves."""
    thetas = samples.thetas(L, sampling, nside)
    thetas = thetas[(thetas > 0) & (thetas < np.pi)]
    return np.concatenate((thetas[:2], thetas[-2:]))


def _valid_entries(L: int, spin: int) -> np.ndarray:
    """Mask of the entries [L - 1 + m, el] of a slice with el >= |spin| and |m| <= el."""
    el = np.arange(L)
    m = np.arange(-L + 1, L)
    return (el[None, :] >= abs(spin)) & (np.abs(m)[:, None] <= el[None, :])


def _check_slices_near_poles(dl: np.ndarray, betas: np.ndarray, L: int, spin: int):
    """Compare slices [L - 1 + m, t, el] with Risbo on their valid entries."""
    valid = _valid_entries(L, spin)
    for t, beta in enumerate(betas):
        np.testing.assert_allclose(
            np.asarray(dl)[:, t][valid],
            _risbo_slice(beta, L, -spin)[valid],
            rtol=0,
            atol=1e-13,
        )


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2, 5, 8, 12, -12])
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_price_mcewen_slices_near_poles(L: int, spin: int, sampling: str):
    """
    Test Price & McEwen slices on the rings nearest the poles against Risbo.

    Near a pole the slice peaks at m = +/-mm and a half recursion continuing past the
    peak amplifies round-off, so that accuracy was lost for |spin| >~ 4 (issue #176).
    """
    nside = L // 2 if sampling == "healpix" else None
    betas = _thetas_nearest_poles(L, sampling, nside)
    dl = recursions.price_mcewen.compute_all_slices(betas, L, spin)
    _check_slices_near_poles(dl, betas, L, spin)


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2, 5, 8, 12, -12])
@pytest.mark.parametrize("sampling", sampling_schemes)
def test_price_mcewen_slices_jax_near_poles(L: int, spin: int, sampling: str):
    """Test JAX Price & McEwen slices on the rings nearest the poles against Risbo."""
    nside = L // 2 if sampling == "healpix" else None
    betas = _thetas_nearest_poles(L, sampling, nside)
    dl = recursions.price_mcewen.compute_all_slices_jax(jnp.asarray(betas), L, spin)
    _check_slices_near_poles(dl, betas, L, spin)


@pytest.mark.parametrize("spin", [-2, 0, 2])
def test_price_mcewen_slices_jax_gradients(spin: int):
    """Test derivatives of JAX Price & McEwen slices with respect to beta."""
    L = 8
    betas = jnp.asarray([0.05, 1.0, 2.0, 3.1])

    def func(betas):
        return recursions.price_mcewen.compute_all_slices_jax(betas, L, spin)

    check_grads(func, (betas,), order=1, modes=("fwd", "rev"))


def test_price_mcewen_slices_jax_gradients_traced_spin():
    """Test beta derivatives of JAX Price & McEwen slices when jit or vmap trace spin."""
    L = 8
    betas = jnp.asarray([0.05, 1.0, 2.0, 3.1])
    spins = jnp.array([-2, 0, 2])

    def loss(betas, spin):
        dl = recursions.price_mcewen.compute_all_slices_jax(betas, L, spin)
        return jnp.sum(dl**2)

    expected = jnp.stack([jax.grad(loss)(betas, int(spin)) for spin in spins])
    compositions = (
        jax.grad(jax.jit(loss))(betas, spins[2]),
        jax.jit(jax.grad(loss))(betas, spins[2]),
    )
    for grad in compositions:
        np.testing.assert_allclose(grad, expected[2], rtol=1e-12, atol=1e-12)
    grads = jax.vmap(jax.grad(loss), in_axes=(None, 0))(betas, spins)
    np.testing.assert_allclose(grads, expected, rtol=1e-12, atol=1e-12)

    def loss_over_spins(betas):
        return jnp.sum(jax.vmap(loss, in_axes=(None, 0))(betas, spins))

    grad = jax.grad(loss_over_spins)(betas)
    np.testing.assert_allclose(grad, expected.sum(0), rtol=1e-12, atol=1e-12)


def test_price_mcewen_single_precision_warning():
    """Test that the JAX recursions warn when double precision is disabled."""
    script = (
        "import jax.numpy as jnp\n"
        "from s2fft.recursions.price_mcewen import compute_all_slices_jax\n"
        "compute_all_slices_jax(jnp.asarray([0.1]), 8, 2)\n"
    )
    result = subprocess.run(  # noqa: S603 (a fixed script)
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_ENABLE_X64": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Wigner-d recursions require double precision" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("sampling", ["mw", "gl"])
def test_price_mcewen_slices_near_poles_high_L(sampling: str):
    """
    Test spin-0 Price & McEwen slices on the rings nearest the poles at L = 128.

    The logarithms of the starting values were accumulated over ~L terms in float64,
    giving errors growing like L^2 log(1/beta) near the poles.
    """
    L = 128
    betas = _thetas_nearest_poles(L, sampling)
    dl = recursions.price_mcewen.compute_all_slices_jax(jnp.asarray(betas), L, 0)
    valid = _valid_entries(L, 0)
    for t, beta in enumerate(betas):
        np.testing.assert_allclose(
            dl[:, t][valid], _risbo_slice(beta, L, 0)[valid], rtol=0, atol=2e-13
        )


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2, 5, 8, 12, -12])
@pytest.mark.parametrize("sampling", ["mw", "gl", "healpix"])
def test_turok_slice_near_poles(L: int, spin: int, sampling: str):
    """Test Turok & Bucher spin slices on the rings nearest the poles against Risbo."""
    nside = L // 2 if sampling == "healpix" else None
    for beta in _thetas_nearest_poles(L, sampling, nside):
        dl_risbo = _risbo_slice(beta, L, -spin)
        for el in range(abs(spin), L):
            m_range = slice(L - 1 - el, L + el)
            dl_turok = recursions.turok.compute_slice(beta, el, L, -spin)
            np.testing.assert_allclose(
                dl_turok[m_range], dl_risbo[m_range, el], rtol=0, atol=1e-13
            )


@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("spin", [0, 2, 5, 8, 12, -12])
@pytest.mark.parametrize("sampling", ["mw", "gl", "healpix"])
def test_turok_slice_jax_near_poles(L: int, spin: int, sampling: str):
    """Test JAX Turok & Bucher spin slices on the rings nearest the poles."""
    nside = L // 2 if sampling == "healpix" else None
    for beta in _thetas_nearest_poles(L, sampling, nside):
        dl_risbo = _risbo_slice(beta, L, -spin)
        for el in range(abs(spin), L):
            m_range = slice(L - 1 - el, L + el)
            dl_turok = recursions.turok_jax.compute_slice(beta, el, L, -spin)
            np.testing.assert_allclose(
                dl_turok[m_range], dl_risbo[m_range, el], rtol=0, atol=1e-13
            )


def test_start_values_scalar():
    """Test the scalar starting values of Turok's slice against the vectorised ones."""
    betas = np.array([1e-3, 0.1, 1.0, np.pi / 2, 2.0, np.pi - 0.1, np.pi - 1e-3])
    el = np.arange(64)
    for mm in [0, 1, -2, 7, -12]:
        mantissas, exponents = recursions.price_mcewen._start_values(betas, el, mm, np)
        scalar = np.array(
            [
                [
                    recursions.price_mcewen._start_values_scalar(beta, ell, mm)
                    for ell in el
                ]
                for beta in betas
            ]
        )
        scalar_mantissas = np.moveaxis(scalar[..., 0], -1, 0)
        scalar_exponents = np.moveaxis(scalar[..., 1], -1, 0)
        np.testing.assert_allclose(
            np.ldexp(scalar_mantissas, (scalar_exponents - exponents).astype(int)),
            mantissas,
            rtol=1e-14,
            atol=0,
        )
