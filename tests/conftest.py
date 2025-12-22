"""Collection of shared fixtures"""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from s2fft.utils import signal_generator

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
    parser.addoption(
        "--cache-directory",
        type=Path,
        default=Path(__file__).parent / "cached-test-data",
        help="Path to access / store cached test data from / to.",
    )
    parser.addoption(
        "--regenerate-cached-data",
        action="store_true",
        help="Regenerate cached test data rather than using stored values.",
    )


def pytest_generate_tests(metafunc):
    for option in ("seed",):
        if option in metafunc.fixturenames:
            metafunc.parametrize(option, metafunc.config.getoption(option))


@pytest.fixture
def cache_directory(request) -> Path:
    return request.config.getoption("cache_directory")


@pytest.fixture
def regenerate_cached_data(request) -> Path:
    return request.config.getoption("regenerate_cached_data")


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def flm_generator(rng):
    return partial(signal_generator.generate_flm, rng)


@pytest.fixture
def flmn_generator(rng):
    return partial(signal_generator.generate_flmn, rng)


@pytest.fixture
def s2fft_to_so3_sampling():
    def so3_sampling(s2fft_sampling):
        if s2fft_sampling.lower() == "mw":
            so3_sampling = "SO3_SAMPLING_MW"
        elif s2fft_sampling.lower() == "mwss":
            so3_sampling = "SO3_SAMPLING_MWSS"
        else:
            raise ValueError(
                f"Sampling scheme sampling={s2fft_sampling} not supported by so3."
            )

        return so3_sampling

    return so3_sampling


def cache_subdirectory_path(cache_directory: Path, subdirectory: str) -> Path:
    cache_subdirectory = cache_directory / subdirectory
    if not cache_subdirectory.exists():
        cache_subdirectory.mkdir(parents=True)
    return cache_subdirectory


def cache_filename(parameters: dict, extension: str = "npz") -> str:
    return "__".join(f"{k}={v}" for k, v in parameters.items()) + "." + extension


@pytest.fixture
def cached_test_case_wrapper(
    cache_directory: Path, regenerate_cached_data: bool, seed: int
) -> Callable[[Callable], Callable]:
    def wrapper(generate_data):
        cache_subdirectory = cache_subdirectory_path(
            cache_directory / generate_data.__module__, generate_data.__qualname__
        )

        def cached_generate_data(**parameters) -> dict[str, np.ndarray]:
            cache_path = cache_subdirectory / cache_filename(
                {"seed": seed} | parameters
            )
            if regenerate_cached_data:
                data = generate_data(**parameters)
                np.savez(cache_path, **data)
                return data
            else:
                return np.load(cache_path)

        return cached_generate_data

    return wrapper
