import jax
import jax.numpy as jnp
import numpy as np
import pyssht as ssht
import pytest
from jax.test_util import check_grads

from s2fft.sampling import s2_samples as samples
from s2fft.transforms.c_backend_spherical import (
    MissingWrapperDependencyError,
    _try_import_module,
)
from s2fft.utils.rotation import generate_rotate_dls, rotate_flms

jax.config.update("jax_enable_x64", True)

L_to_test = [6, 8, 10]
angles_to_test = [np.pi / 2, np.pi / 6]


def test_flm_reindexing_functions(flm_generator):
    L = 16
    flm_2d = flm_generator(L=L, spin=0, reality=False)

    flm_1d = samples.flm_2d_to_1d(flm_2d, L)

    assert len(flm_1d.shape) == 1

    flm_2d_check = samples.flm_1d_to_2d(flm_1d, L)

    assert len(flm_2d_check.shape) == 2
    np.testing.assert_allclose(flm_2d, flm_2d_check, atol=1e-14)


def test_flm_reindexing_functions_healpix(flm_generator):
    L = 16
    flm_2d = flm_generator(L=L, spin=0, reality=True)
    flm_hp = samples.flm_2d_to_hp(flm_2d, L)

    flm_2d_check = samples.flm_hp_to_2d(flm_hp, L)

    assert len(flm_2d_check.shape) == 2
    np.testing.assert_allclose(flm_2d, flm_2d_check, atol=1e-14)


def test_flm_reindexing_exceptions(flm_generator):
    L = 16
    spin = 0

    flm_2d = flm_generator(L=L, spin=spin, reality=False)
    flm_1d = samples.flm_2d_to_1d(flm_2d, L)
    flm_3d = np.zeros((1, 1, 1))

    with pytest.raises(ValueError):
        samples.flm_2d_to_1d(flm_1d, L)

    with pytest.raises(ValueError):
        samples.flm_2d_to_1d(flm_3d, L)

    with pytest.raises(ValueError):
        samples.flm_1d_to_2d(flm_2d, L)

    with pytest.raises(ValueError):
        samples.flm_1d_to_2d(flm_3d, L)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("alpha", angles_to_test)
@pytest.mark.parametrize("beta", angles_to_test)
@pytest.mark.parametrize("gamma", angles_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_rotate_flms(flm_generator, L: int, alpha: float, beta: float, gamma: float):
    flm = flm_generator(L=L)
    rot = (alpha, beta, gamma)
    flm_1d = samples.flm_2d_to_1d(flm, L)

    flm_rot_ssht = samples.flm_1d_to_2d(
        ssht.rotate_flms(flm_1d, alpha, beta, gamma, L), L
    )
    flm_rot_s2fft = rotate_flms(flm, L, rot)

    np.testing.assert_allclose(flm_rot_ssht, flm_rot_s2fft, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("alpha", angles_to_test)
@pytest.mark.parametrize("beta", angles_to_test)
@pytest.mark.parametrize("gamma", angles_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_rotate_flms_precompute_dls(
    flm_generator, L: int, alpha: float, beta: float, gamma: float
):
    dl = generate_rotate_dls(L, beta)
    flm = flm_generator(L=L)
    rot = (alpha, beta, gamma)
    flm_1d = samples.flm_2d_to_1d(flm, L)

    flm_rot_ssht = samples.flm_1d_to_2d(
        ssht.rotate_flms(flm_1d, alpha, beta, gamma, L), L
    )
    flm_rot_s2fft = rotate_flms(flm, L, rot, dl)

    np.testing.assert_allclose(flm_rot_ssht, flm_rot_s2fft, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("alpha", angles_to_test)
@pytest.mark.parametrize("beta", angles_to_test)
@pytest.mark.parametrize("gamma", angles_to_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_rotate_flms_gradients(
    flm_generator, L: int, alpha: float, beta: float, gamma: float
):
    flm_start = flm_generator(L=L)

    rot = (alpha, beta, gamma)
    flm_target = rotate_flms(flm_start, L, (0.1, 0.1, 0.1))

    def func(flm):
        flm_rot = rotate_flms(flm, L, rot)
        return jnp.sum(jnp.abs(flm_rot - flm_target))

    check_grads(func, (flm_start,), order=1, modes=("rev"))


def test_try_import_module():
    # Use an intentionally long and unlikely to clash module name
    module_name = "_a_random_module_name_that_should_not_exist"
    with pytest.raises(
        MissingWrapperDependencyError, match="requires {module_name} to be installed"
    ):
        _try_import_module(module_name)
