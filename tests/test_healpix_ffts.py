from collections.abc import Callable

import jax
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

nside_to_test = [4, 5]
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
