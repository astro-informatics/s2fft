from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from packaging.version import Version as _Version

from s2fft.utils.healpix_ffts import (
    healpix_fft_cuda,
    healpix_fft_jax,
    healpix_fft_numpy,
    healpix_ifft_cuda,
    healpix_ifft_jax,
    healpix_ifft_numpy,
)

if _Version(jax.__version__) < _Version("0.4.32"):
    from jax.lib.xla_bridge import get_backend
else:
    from jax.extend.backend import get_backend

gpu_available = get_backend().platform == "gpu"

jax.config.update("jax_enable_x64", True)

nside_to_test = [8, 16]
reality_to_test = [False, True]


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_healpix_fft_jax_numpy_consistency(
    cached_healpy_test_case: Callable, nside, reality
):
    L = 2 * nside
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    assert np.allclose(
        healpix_fft_numpy(test_data["f_hp"], L, nside, reality),
        healpix_fft_jax(test_data["f_hp"], L, nside, reality),
    )


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_healpix_ifft_jax_numpy_consistency(
    cached_healpy_test_case: Callable, nside, reality
):
    L = 2 * nside
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    ftm = healpix_fft_numpy(test_data["f_hp"], L, nside, reality)
    ftm_copy = np.copy(ftm)
    # Test consistency
    assert np.allclose(
        healpix_ifft_numpy(ftm, L, nside, reality),
        healpix_ifft_jax(ftm_copy, L, nside, reality),
    )


@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_fft_cuda(cached_healpy_test_case: Callable, nside):
    L = 2 * nside
    reality = False
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    # Test consistency
    assert_allclose(
        healpix_fft_jax(test_data["f_hp"], L, nside, reality),
        healpix_fft_cuda(test_data["f_hp"], L, nside, reality),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_ifft_cuda(cached_healpy_test_case: Callable, nside):
    L = 2 * nside
    reality = False
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    ftm = healpix_fft_jax(test_data["f_hp"], L, nside, reality)
    # Test consistency
    assert_allclose(
        healpix_ifft_jax(ftm, L, nside, reality).flatten(),
        healpix_ifft_cuda(ftm, L, nside, reality).flatten(),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_fft_cuda_no_input_mutation(cached_healpy_test_case, nside):
    L = 2 * nside
    reality = False
    test_data = cached_healpy_test_case(L=L, nside=nside, reality=reality)
    f = test_data["f_hp"]
    f_copy = f.copy()

    # Forward: input f must not be corrupted
    ftm_1 = healpix_fft_cuda(f, L, nside, False)
    assert_allclose(f, f_copy, atol=0, rtol=0, err_msg="forward call 1 corrupted input")

    ftm_2 = healpix_fft_cuda(f, L, nside, False)
    assert_allclose(f, f_copy, atol=0, rtol=0, err_msg="forward call 2 corrupted input")

    assert_allclose(
        ftm_1, ftm_2, atol=0, rtol=0, err_msg="forward results differ between calls"
    )

    # Backward: input ftm must not be corrupted
    ftm = healpix_fft_cuda(f, L, nside, False)
    ftm_copy = ftm.copy()

    f_1 = healpix_ifft_cuda(ftm, L, nside, False)
    assert_allclose(
        ftm, ftm_copy, atol=0, rtol=0, err_msg="backward call 1 corrupted input"
    )

    f_2 = healpix_ifft_cuda(ftm, L, nside, False)
    assert_allclose(
        ftm, ftm_copy, atol=0, rtol=0, err_msg="backward call 2 corrupted input"
    )

    assert_allclose(
        f_1, f_2, atol=0, rtol=0, err_msg="backward results differ between calls"
    )


@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_fft_cuda_transforms(cached_healpy_test_case, nside):
    L = 2 * nside

    f_stacked = [
        cached_healpy_test_case(L=L, nside=nside, reality=False)["f_hp"]
        for _ in range(3)
    ]

    f_stacked = (
        jnp.stack(f_stacked, axis=0)
        + jax.random.normal(jax.random.PRNGKey(0), (3,)).reshape(-1, 1) * 1e-6
    )

    def healpix_jax(f):
        return healpix_fft_jax(f, L, nside, False).real

    def healpix_cuda(f):
        return healpix_fft_cuda(f, L, nside, False).real

    f = f_stacked[0]
    # Test VMAP
    assert_allclose(
        jax.vmap(healpix_jax)(f_stacked),
        jax.vmap(healpix_cuda)(f_stacked),
        atol=1e-7,
        rtol=1e-7,
    )
    # test jacfwd
    assert_allclose(
        jax.jacfwd(healpix_jax)(f.real),
        jax.jacfwd(healpix_cuda)(f.real),
        atol=1e-7,
        rtol=1e-7,
    )
    # test jacrev
    assert_allclose(
        jax.jacrev(healpix_jax)(f.real),
        jax.jacrev(healpix_cuda)(f.real),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@pytest.mark.parametrize("nside", nside_to_test)
def test_healpix_ifft_cuda_transforms(cached_healpy_test_case, nside):
    L = 2 * nside

    test_data = cached_healpy_test_case(L=L, nside=nside, reality=False)
    ftm = healpix_fft_jax(test_data["f_hp"], L, nside, False)
    ftm_stacked = [ftm for _ in range(3)]
    ftm_stacked = (
        jnp.stack(ftm_stacked, axis=0)
        + jax.random.normal(jax.random.PRNGKey(0), (3,)).reshape(-1, 1, 1) * 1e-6
    )
    ftm = ftm_stacked[0].real

    def healpix_inv_jax(ftm):
        return healpix_ifft_jax(ftm, L, nside, False).real

    def healpix_inv_cuda(ftm):
        return healpix_ifft_cuda(ftm, L, nside, False).real

    # Test VMAP
    assert_allclose(
        jax.vmap(healpix_inv_jax)(ftm_stacked),
        jax.vmap(healpix_inv_cuda)(ftm_stacked),
        atol=1e-7,
        rtol=1e-7,
    )
    # test jacfwd
    assert_allclose(
        jax.jacfwd(healpix_inv_jax)(ftm.real),
        jax.jacfwd(healpix_inv_cuda)(ftm.real),
        atol=1e-7,
        rtol=1e-7,
    )
    # test jacrev
    assert_allclose(
        jax.jacrev(healpix_inv_jax)(ftm.real),
        jax.jacrev(healpix_inv_cuda)(ftm.real),
        atol=1e-7,
        rtol=1e-7,
    )
