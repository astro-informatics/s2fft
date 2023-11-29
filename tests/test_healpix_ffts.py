import numpy as np
import pytest
from jax import config
from s2fft.utils.healpix_ffts import (
    healpix_fft_jax,
    healpix_fft_numpy,
    healpix_ifft_jax,
    healpix_ifft_numpy,
)


config.update("jax_enable_x64", True)


@pytest.mark.parametrize("L", (32, 64))
@pytest.mark.parametrize("nside", (4, 8, 16))
@pytest.mark.parametrize("reality", (True, False))
def test_healpix_fft_jax_numpy_consistency(rng, L, nside, reality):
    f = rng.standard_normal(size=12 * nside**2)
    assert np.allclose(
        healpix_fft_numpy(f, L, nside, reality), healpix_fft_jax(f, L, nside, reality)
    )


@pytest.mark.parametrize("L", (32, 64))
@pytest.mark.parametrize("nside", (4, 8, 16))
@pytest.mark.parametrize("reality", (True, False))
def test_healpix_ifft_jax_numpy_consistency(rng, L, nside, reality):
    ftm = healpix_fft_numpy(
        rng.standard_normal(size=12 * nside**2), L, nside, reality
    )
    assert np.allclose(
        healpix_ifft_numpy(ftm, L, nside, reality),
        healpix_ifft_jax(ftm, L, nside, reality),
    )
