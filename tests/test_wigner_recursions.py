from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
