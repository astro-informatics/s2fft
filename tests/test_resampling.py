import pytest
import numpy as np
import s2fft as s2f


def test_periodic_extension_invalid_sampling():

    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError) as e:
        s2f.resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="healpix")

    with pytest.raises(ValueError) as e:
        s2f.resampling.periodic_extension(f_dummy, L=5, spin=0, sampling="dh")


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_periodic_extension_mwss(flm_generator, L: int, spin_reality):

    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling="mwss")

    f_ext = s2f.resampling.periodic_extension(f, L, spin, sampling="mwss")

    f_ext_check = s2f.resampling.periodic_extension_spatial_mwss(f, L, spin)

    np.testing.assert_allclose(f_ext, f_ext_check, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mwss_upsample_downsample(flm_generator, L: int, spin_reality):

    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling="mwss")

    f_ext = s2f.resampling.periodic_extension_spatial_mwss(f, L, spin)

    f_ext_up = s2f.resampling.upsample_by_two_mwss_ext(f_ext, L)

    f_ext_up_down = s2f.resampling.downsample_by_two_mwss(f_ext_up, 2 * L)

    np.testing.assert_allclose(f_ext, f_ext_up_down, atol=1e-10)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize("sampling", ["mw", "mwss"])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_unextend(flm_generator, L: int, sampling: str, spin_reality):

    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f = s2f.transform.inverse_sov_fft(flm, L, spin, sampling=sampling)

    f_ext = s2f.resampling.periodic_extension(f, L, spin, sampling=sampling)

    f_unext = s2f.resampling.unextend(f_ext, L, sampling)

    np.testing.assert_allclose(f, f_unext, atol=1e-10)


def test_resampling_exceptions():

    f_dummy = np.zeros((2, 2), dtype=np.complex128)

    with pytest.raises(ValueError) as e:
        L_odd = 3
        s2f.resampling.downsample_by_two_mwss(f_dummy, L_odd)

    with pytest.raises(ValueError) as e:
        s2f.resampling.unextend(f_dummy, L=5, sampling="healpix")

    with pytest.raises(ValueError) as e:
        s2f.resampling.unextend(f_dummy, L=5, sampling="dh")

    with pytest.raises(ValueError) as e:
        # f_ext has wrong shape
        s2f.resampling.unextend(f_dummy, L=5, sampling="mw")

    with pytest.raises(ValueError) as e:
        s2f.resampling.mw_to_mwss_phi(f_dummy, L=5)

    L = 5
    nphi_mw = s2f.samples.nphi_equiang(L, sampling="mw")
    ntheta_mw = s2f.samples.ntheta(L, sampling="mw")
    with pytest.raises(ValueError) as e:
        f_dummy = np.zeros((ntheta_mw + 1, nphi_mw), dtype=np.complex128)
        s2f.resampling.mw_to_mwss_theta(f_dummy, L)

    with pytest.raises(ValueError) as e:
        f_dummy = np.zeros((ntheta_mw, nphi_mw + 1), dtype=np.complex128)
        s2f.resampling.mw_to_mwss_theta(f_dummy, L)


@pytest.mark.parametrize("L", [5])
@pytest.mark.parametrize(
    "spin_reality", [(0, True), (0, False), (1, False), (2, False)]
)
def test_mw_to_mwss_theta(flm_generator, L: int, spin_reality):

    (spin, reality) = spin_reality
    flm = flm_generator(L=L, spin=spin, reality=reality)
    f_mw = s2f.transform.inverse_sov_fft(flm, L, spin, sampling="mw")
    f_mwss = s2f.transform.inverse_sov_fft(flm, L, spin, sampling="mwss")

    f_mwss_converted = s2f.resampling.mw_to_mwss(f_mw, L, spin)

    np.testing.assert_allclose(f_mwss, f_mwss_converted, atol=1e-10)
