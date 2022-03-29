import pytest
import numpy as np
import s2fft as s2f
import pyssht as ssht


@pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh"])
def test_transform_inverse_direct(L: int, spin: int, sampling: str):

    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    f_check = ssht.inverse(flm, L, Method=sampling.upper(), Spin=spin, Reality=False)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", [15])
@pytest.mark.parametrize("spin", [0])
@pytest.mark.parametrize("sampling", ["mw"])
def test_transform_inverse_sov(L: int, spin: int, sampling: str):

    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    f = s2f.transform.inverse_sov(flm, L, spin, sampling)

    f_check = ssht.inverse(flm, L, Method=sampling.upper(), Spin=spin, Reality=False)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", [15])
@pytest.mark.parametrize("spin", [0])
@pytest.mark.parametrize("sampling", ["mw"])
def test_transform_inverse_sov_fft(L: int, spin: int, sampling: str):

    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling)

    f_check = ssht.inverse(flm, L, Method=sampling.upper(), Spin=spin, Reality=False)

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.skip(reason="Temporarily skipped for faster development")
@pytest.mark.parametrize("L", [15, 16])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("sampling", ["dh"])
def test_transform_forward_direct(L: int, spin: int, sampling: str):

    # TODO: move this and potentially do better
    np.random.seed(2)

    # Create bandlimited signal
    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    # Zero harmonic coefficient with el < abs(spin)
    el = np.abs(spin) - 1
    m = el
    ind = s2f.sampling.elm2ind(el, m)
    flm[0 : ind + 1] = 0.0

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    flm_recov = s2f.transform.forward_direct(f, L, spin, sampling)

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)


@pytest.mark.parametrize("L", [15])
@pytest.mark.parametrize("spin", [0])
@pytest.mark.parametrize("sampling", ["dh"])
def test_transform_forward_sov(L: int, spin: int, sampling: str):

    # TODO: move this and potentially do better
    np.random.seed(2)

    # Create bandlimited signal
    ncoeff = s2f.sampling.ncoeff(L)
    flm = np.zeros(ncoeff, dtype=np.complex128)
    flm = np.random.rand(ncoeff) + 1j * np.random.rand(ncoeff)

    # Zero harmonic coefficient with el < abs(spin)
    el = np.abs(spin) - 1
    m = el
    ind = s2f.sampling.elm2ind(el, m)
    flm[0 : ind + 1] = 0.0

    f = s2f.transform.inverse_direct(flm, L, spin, sampling)

    flm_recov = s2f.transform.forward_sov_fft(f, L, spin, sampling)

    np.testing.assert_allclose(flm, flm_recov, atol=1e-14)
