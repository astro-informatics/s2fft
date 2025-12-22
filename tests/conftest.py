"""Collection of shared fixtures"""

import json

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, ParamSpec, TypeAlias

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
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def flm_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flm, rng)


@pytest.fixture
def flmn_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flmn, rng)


@pytest.fixture
def s2fft_to_so3_sampling() -> Callable[[str], str]:
    def so3_sampling(s2fft_sampling: str) -> str:
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


def cache_filename(parameters: dict, extension: str) -> str:
    return "__".join(f"{k}={v}" for k, v in parameters.items()) + "." + extension


P = ParamSpec("P")
TestData: TypeAlias = Mapping[str, Any]


class TestDataFormat(NamedTuple):
    extension: str
    load: Callable[[Path], TestData]
    save: Callable[[Path, TestData], None]


def npz_load(path: Path) -> TestData:
    return np.load(path)


def npz_save(path: Path, data: TestData) -> None:
    return np.savez(path, **data)


def json_load(path: Path) -> TestData:
    with path.open("r") as f:
        return json.load(f)


def json_save(path: Path, data: TestData) -> None:
    with path.open("w") as f:
        json.dump(data, f)


TEST_DATA_FORMATS = {
    "npz": TestDataFormat("npz", npz_load, npz_save),
    "json": TestDataFormat("json", json_load, json_save),
}


@pytest.fixture
def cached_test_case_wrapper(
    cache_directory: Path, regenerate_cached_data: bool, seed: int
) -> Callable[[Callable[P, TestData], str], Callable[P, TestData]]:
    """Fixture for caching generated test data to file.

    Returns a wrapper function which when applied to a function `generate_data` which
    generates test data will call the `generate_data` function if
    `regenerate_cached_data` is True and cache the generated test data in a file named
    according to the keyword arguments to `generate_data` under a module / function
    specific subdirectory in `cache_directory`, and otherwise will attempt to load
    previously cached data from `cache_directory`. The format the cached data is written
    to and read from is specified by `format`.
    """

    def wrapper(
        generate_data: Callable[P, TestData], format: str
    ) -> Callable[P, TestData]:
        data_format = TEST_DATA_FORMATS[format]
        cache_subdirectory = cache_subdirectory_path(
            cache_directory / generate_data.__module__, generate_data.__qualname__
        )

        def cached_generate_data(*args: P.args, **kwargs: P.kwargs) -> TestData:
            cache_path = cache_subdirectory / cache_filename(
                {"seed": seed} | kwargs, data_format.extension
            )
            if regenerate_cached_data:
                data = generate_data(*args, **kwargs)
                data_format.save(cache_path, data)
                return data
            else:
                return data_format.load(cache_path)

        return cached_generate_data

    return wrapper


@pytest.fixture
def cached_so3_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flmn_generator: Callable[..., np.ndarray],
    s2fft_to_so3_sampling: Callable[[str], str],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, N: int, L_lower: int, sampling: str, reality: bool
    ) -> dict[str, np.ndarray]:
        import so3

        from s2fft.sampling import so3_samples

        flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)

        so3_parameters = so3.create_parameter_dict(
            L=L,
            N=N,
            L0=L_lower,
            sampling_scheme_str=s2fft_to_so3_sampling(sampling),
            reality=False,
        )

        f_so3 = so3.inverse(so3_samples.flmn_3d_to_1d(flmn, L, N), so3_parameters)
        flmn_so3 = so3_samples.flmn_1d_to_3d(so3.forward(f_so3, so3_parameters), L, N)

        return {"flmn": flmn, "f_so3": f_so3, "flmn_so3": flmn_so3}

    return cached_test_case_wrapper(generate_data, "npz")
