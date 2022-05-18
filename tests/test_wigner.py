import pytest
import numpy as np
import jax.numpy as jnp
import s2fft.wigner as wigner
import s2fft.samples as samples
import pyssht as ssht

from jax.config import config

config.update("jax_enable_x64", True)

L_to_test = [8, 16, 32]
spin_to_test = np.arange(-3, 3)


def test_trapani_with_ssht():
    """Test Trapani computation against ssht"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compute using SSHT.
    beta = np.pi / 2.0
    dl_array = ssht.generate_dl(beta, L)

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = wigner.trapani.init(dl, L)
    for el in range(1, L):
        dl = wigner.trapani.compute_full(dl, L, el)
        np.testing.assert_allclose(dl_array[el, :, :], dl, atol=1e-10)


def test_trapani_vectorized():
    """Test vectorized Trapani computation"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = wigner.trapani.init(dl, L)
    dl_vect = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl_vect = wigner.trapani.init(dl_vect, L)
    for el in range(1, L):
        dl = wigner.trapani.compute_full(dl, L, el)
        dl_vect = wigner.trapani.compute_full_vectorized(dl_vect, L, el)
        np.testing.assert_allclose(
            dl[-el + (L - 1) : el + (L - 1) + 1, -el + (L - 1) : el + (L - 1) + 1],
            dl_vect[-el + (L - 1) : el + (L - 1) + 1, -el + (L - 1) : el + (L - 1) + 1],
            atol=1e-10,
        )


def test_trapani_jax():
    """Test JAX Trapani computation"""

    # Test all dl(pi/2) terms up to L.
    L = 32

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    dl = wigner.trapani.init(dl, L)
    dl_jax = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl_jax = wigner.trapani.init_jax(dl_jax, L)
    for el in range(1, L):
        dl = wigner.trapani.compute_full_vectorized(dl, L, el)
        dl_jax = wigner.trapani.compute_full_jax(dl_jax, L, el)
        np.testing.assert_allclose(
            dl[-el + (L - 1) : el + (L - 1) + 1, -el + (L - 1) : el + (L - 1) + 1],
            dl_jax[-el + (L - 1) : el + (L - 1) + 1, -el + (L - 1) : el + (L - 1) + 1],
            atol=1e-10,
        )


def test_trapani_checks():

    # TODO

    # Check throws exception if arguments wrong

    # Check throws exception if don't init

    return


def test_risbo_with_ssht():
    """Test Risbo computation against ssht"""

    # Test all dl(pi/2) terms up to L.
    L = 10

    # Compute using SSHT.
    beta = np.pi / 2.0
    dl_array = ssht.generate_dl(beta, L)

    # Compare to routines in SSHT, which have been validated extensively.
    dl = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)
    # dl = wigner.trapani.init(dl, L)
    for el in range(0, L):
        dl = wigner.risbo.compute_full(dl, beta, L, el)
        np.testing.assert_allclose(dl_array[el, :, :], dl, atol=1e-15)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "healpix"])
def test_turok_with_ssht(L: int, sampling: str):
    """Test Turok computation against ssht"""

    # Test all dl() terms up to L.
    betas = samples.thetas(L, sampling, int(L / 2))

    # Compute using SSHT.
    for beta in betas:
        dl_array = ssht.generate_dl(beta, L)

        for el in range(L):

            dl_turok = wigner.turok.compute_full(beta, el, L)

            np.testing.assert_allclose(dl_turok, dl_array[el], atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "healpix"])
def test_turok_slice_with_ssht(L: int, spin: int, sampling: str):
    """Test Turok spin slice computation against ssht"""

    # Test all dl() terms up to L.
    betas = samples.thetas(L, sampling, int(L / 2))

    # Compute using SSHT.
    for beta in betas:
        dl_array = ssht.generate_dl(beta, L)

        for el in range(L):
            if el >= np.abs(spin):

                dl_turok = np.flip(wigner.turok.compute_slice(beta, el, L, -spin))

                np.testing.assert_allclose(
                    dl_turok, dl_array[el][L - 1 + spin], atol=1e-10, rtol=1e-12
                )


def test_turok_exceptions():
    L = 10

    with pytest.raises(ValueError) as e:
        wigner.turok.compute_full(np.pi / 2, L, L)

    with pytest.raises(ValueError) as e:
        wigner.turok.compute_slice(beta=np.pi / 2, el=L - 1, L=L, mm=L)

    with pytest.raises(ValueError) as e:
        wigner.turok.compute_slice(beta=np.pi / 2, el=L, L=L, mm=0)

    with pytest.raises(ValueError) as e:
        wigner.turok.compute_slice(beta=np.pi / 2, el=L - 1, L=L, mm=0)
