import pytest
import numpy as np
from s2fft.utils import resampling
from s2fft.base_transforms import spherical


def test_periodic_extension_invalid_sampling():
    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError) as e:
        resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="healpix")

    with pytest.raises(ValueError) as e:
        resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="dh")


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_periodic_extension_mwss(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling="mwss")
    f = np.expand_dims(f, 0)

    f_ext = resampling.periodic_extension(f, L, spin, sampling="mwss")

    f_ext_check = resampling.periodic_extension_spatial_mwss(f, L, spin)

    np.testing.assert_allclose(f_ext, f_ext_check, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mwss_upsample_downsample(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling="mwss")
    f = np.expand_dims(f, 0)

    f_ext = resampling.periodic_extension_spatial_mwss(f, L, spin)

    f_ext_up = resampling.upsample_by_two_mwss_ext(f_ext, L)

    f_ext_up_down = resampling.downsample_by_two_mwss(f_ext_up, 2 * L)

    np.testing.assert_allclose(f_ext, f_ext_up_down, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_unextend(flm_generator, L: int, sampling: str, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = spherical.inverse(flm, L, spin, sampling=sampling)
    f = np.expand_dims(f, 0)
    f_ext = resampling.periodic_extension(f, L, spin, sampling=sampling)

    f_unext = resampling.unextend(f_ext, L, sampling)

    np.testing.assert_allclose(f, f_unext, atol=1e-10)


def test_resampling_exceptions():
    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError) as e:
        L_odd = 3
        resampling.downsample_by_two_mwss(f_dummy, L_odd)

    with pytest.raises(ValueError) as e:
        resampling.unextend(f_dummy, L=5, sampling="healpix")

    with pytest.raises(ValueError) as e:
        resampling.unextend(f_dummy, L=5, sampling="dh")

    with pytest.raises(ValueError) as e:
        # f_ext has wrong shape
        resampling.unextend(f_dummy, L=5, sampling="mw")

    with pytest.raises(ValueError) as e:
        resampling.mw_to_mwss_phi(f_dummy, L=5)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mw_to_mwss_theta(flm_generator, L: int, spin_reality):
    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_mw = spherical.inverse(flm, L, spin, sampling="mw")
    f_mwss = spherical.inverse(flm, L, spin, sampling="mwss")

    f_mwss_converted = resampling.mw_to_mwss(f_mw, L, spin)

    np.testing.assert_allclose(f_mwss, f_mwss_converted, atol=1e-10)
