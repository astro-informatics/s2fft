import numpy as np
import healpy as hp
import pytest
from jax import config
from s2fft.sampling import s2_samples as samples
from s2fft.utils.healpix_ffts import (
    healpix_fft_jax,
    healpix_fft_numpy,
    healpix_ifft_jax,
    healpix_ifft_numpy,
)


config.update("jax_enable_x64", True)


nside_to_test = [4, 5]
reality_to_test = [False, True]


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_healpix_fft_jax_numpy_consistency(flm_generator, nside, reality):
    L = 2 * nside
    # Generate a random bandlimited signal
    flm = flm_generator(L=L, reality=reality)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)
    # Test consistency
    assert np.allclose(
        healpix_fft_numpy(f, L, nside, reality), healpix_fft_jax(f, L, nside, reality)
    )


@pytest.mark.parametrize("nside", nside_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
def test_healpix_ifft_jax_numpy_consistency(flm_generator, nside, reality):
    L = 2 * nside
    # Generate a random bandlimited signal
    flm = flm_generator(L=L, reality=reality)
    flm_hp = samples.flm_2d_to_hp(flm, L)
    f = hp.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)
    ftm = healpix_fft_numpy(f, L, nside, reality)
    ftm_copy = np.copy(ftm)
    # Test consistency
    assert np.allclose(
        healpix_ifft_numpy(ftm, L, nside, reality),
        healpix_ifft_jax(ftm_copy, L, nside, reality),
    )
