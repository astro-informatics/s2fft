"""Collection of shared fixtures"""
from functools import partial
import pytest
import numpy as np
import s2fft as s2f


DEFAULT_SEED = 8966433580120847635


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help=(
            "Seed(s) to use for random number generator fixture rng in tests. If "
            "multiple seeds are passed tests depending on rng will be run for all "
            "seeds specified."
        ),
    )


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"))


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def flm_generator(rng):
    return partial(s2f.utils.generate_flm, rng)
