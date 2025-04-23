import jax
import numpy as np
import pytest

from s2fft.precompute_transforms import construct as c
from s2fft.precompute_transforms import custom_ops as ops
from s2fft.precompute_transforms import fourier_wigner as fw

jax.config.update("jax_enable_x64", True)

# Test cases
L_to_test = [16]
N_to_test = [16]
sampling_schemes = ["mw", "mwss"]
methods_to_test = ["numpy", "jax"]

# Test tolerance
atol = 1e-12


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("method", methods_to_test)
def test_custom_forward_from_s2(
    flmn_generator, L: int, N: int, sampling: str, method: str
):
    # GENERATE MOCK SIGNAL
    flmn = flmn_generator(L=L, N=N)
    precomps = c.fourier_wigner_kernel(L)
    f = fw.inverse_transform(flmn, L, N, precomps, False, sampling)
    spins = -np.arange(-N + 1, N)

    # FUNCTION SWITCH
    func = ops.s2_to_wigner_subset_jax if method == "jax" else ops.s2_to_wigner_subset

    # INVERT FFT OVER GAMMA
    fn = np.fft.fft(f, axis=-3, norm="forward")
    fn = np.fft.fftshift(fn, axes=-3)

    # CREATE CORRECT SHAPE (BATCH: 1, CHANNELS: 1)
    fn = fn.reshape((1,) + fn.shape + (1,))

    # TEST: ALL UNIQUE SPINS
    flmn_test = func(fn, spins, precomps, L, sampling)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test), atol=atol)

    # TEST: A SINGLE SPIN
    flmn_test = func(fn[:, [0]], spins[[0]], precomps, L, sampling)
    np.testing.assert_allclose(flmn[0], np.squeeze(flmn_test), atol=atol)

    # TEST: SUBSET OF SPINS
    flmn_test = func(fn[:, ::2], spins[::2], precomps, L, sampling)
    np.testing.assert_allclose(flmn[::2], np.squeeze(flmn_test), atol=atol)

    # TEST: REPEATED SPINS
    fn_repeat = np.concatenate([fn, fn], axis=1)
    spins_repeat = np.concatenate([spins, spins])
    flmn_test = func(fn_repeat, spins_repeat, precomps, L, sampling)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test[:, : len(spins)]), atol=atol)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test[:, len(spins) :]), atol=atol)

    # TEST: SIMULATED BATCHING
    fnb = np.concatenate([fn, fn], axis=0)
    flmn_test = func(fnb, spins, precomps, L, sampling)
    for b in range(2):
        np.testing.assert_allclose(flmn, np.squeeze(flmn_test[b]), atol=atol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("method", methods_to_test)
def test_custom_forward_from_so3(
    flmn_generator, L: int, N: int, sampling: str, method: str
):
    # GENERATE MOCK SIGNAL
    flmn = flmn_generator(L=L, N=N)
    precomps = c.fourier_wigner_kernel(L)
    f = fw.inverse_transform(flmn, L, N, precomps, False, sampling)
    spins = -np.arange(-N + 1, N)

    # FUNCTION SWITCH
    func = ops.so3_to_wigner_subset_jax if method == "jax" else ops.so3_to_wigner_subset

    # CREATE CORRECT SHAPE (BATCH: 1, CHANNELS: 1)
    f = f.reshape((1,) + f.shape + (1,))

    # TEST: ALL UNIQUE SPINS
    flmn_test = func(f, spins, precomps, L, N, sampling)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test), atol=atol)

    # TEST: A SINGLE SPIN
    flmn_test = func(f, spins[[0]], precomps, L, N, sampling)
    np.testing.assert_allclose(flmn[0], np.squeeze(flmn_test), atol=atol)

    # TEST: SUBSET OF SPINS
    flmn_test = func(f, spins[::2], precomps, L, N, sampling)
    np.testing.assert_allclose(flmn[::2], np.squeeze(flmn_test), atol=atol)

    # TEST: REPEATED SPINS
    spins_repeat = np.concatenate([spins, spins])
    flmn_test = func(f, spins_repeat, precomps, L, N, sampling)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test[:, : len(spins)]), atol=atol)
    np.testing.assert_allclose(flmn, np.squeeze(flmn_test[:, len(spins) :]), atol=atol)

    # TEST: SIMULATED BATCHING
    fb = np.concatenate([f, f], axis=0)
    flmn_test = func(fb, spins, precomps, L, N, sampling)
    for b in range(2):
        np.testing.assert_allclose(flmn, np.squeeze(flmn_test[b]), atol=atol)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("sampling", sampling_schemes)
@pytest.mark.parametrize("method", methods_to_test)
def test_custom_inverse_to_s2(
    flmn_generator, L: int, N: int, sampling: str, method: str
):
    # GENERATE MOCK SIGNAL
    flmn = flmn_generator(L=L, N=N)
    precomps = c.fourier_wigner_kernel(L)
    f = fw.inverse_transform(flmn, L, N, precomps, False, sampling)
    spins = -np.arange(-N + 1, N)

    # FUNCTION SWITCH
    func = ops.wigner_subset_to_s2_jax if method == "jax" else ops.wigner_subset_to_s2

    # INVERT FFT OVER GAMMA
    fn = np.fft.fft(f, axis=-3, norm="forward")
    fn = np.fft.fftshift(fn, axes=-3)

    # CREATE CORRECT SHAPE (BATCH: 1, CHANNELS: 1)
    flmn = flmn.reshape((1,) + flmn.shape + (1,))

    # TEST: ALL UNIQUE SPINS
    f_test = func(flmn, spins, precomps, L, sampling)
    np.testing.assert_allclose(fn, np.squeeze(f_test), atol=atol)

    # TEST: A SINGLE SPIN
    f_test = func(flmn[:, [0]], spins[[0]], precomps, L, sampling)
    np.testing.assert_allclose(fn[0], np.squeeze(f_test), atol=atol)

    # TEST: SUBSET OF SPINS
    f_test = func(flmn[:, ::2], spins[::2], precomps, L, sampling)
    np.testing.assert_allclose(fn[::2], np.squeeze(f_test), atol=atol)

    # TEST: REPEATED SPINS
    flmn_repeat = np.concatenate([flmn, flmn], axis=1)
    spins_repeat = np.concatenate([spins, spins])
    f_test = func(flmn_repeat, spins_repeat, precomps, L, sampling)
    np.testing.assert_allclose(fn, np.squeeze(f_test[:, : len(spins)]), atol=atol)
    np.testing.assert_allclose(fn, np.squeeze(f_test[:, len(spins) :]), atol=atol)

    # TEST: SIMULATED BATCHING
    flmnb = np.concatenate([flmn, flmn], axis=0)
    f_test = func(flmnb, spins, precomps, L, sampling)
    for b in range(2):
        np.testing.assert_allclose(fn, np.squeeze(f_test[b]), atol=atol)
