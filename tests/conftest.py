"""Collection of shared fixtures"""

import inspect
import json
from collections.abc import Callable, Mapping
from functools import partial, wraps
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
        "--use-cache",
        action="store_true",
        help="Use cached test data rather than generating dynamically.",
    )
    parser.addoption(
        "--update-cache",
        action="store_true",
        help="Update cached test data values.",
    )


def pytest_generate_tests(metafunc):
    for option in ("seed",):
        if option in metafunc.fixturenames:
            metafunc.parametrize(option, metafunc.config.getoption(option))


def pytest_collection_modifyitems(items):
    for item in items:
        if "cached_test_case_wrapper" in getattr(item, "fixturenames", ()):
            item.add_marker("uses_cached_data")


@pytest.fixture
def cache_directory(request) -> Path:
    return request.config.getoption("cache_directory")


@pytest.fixture
def use_cache(request) -> Path:
    return request.config.getoption("use_cache")


@pytest.fixture
def update_cache(request) -> Path:
    return request.config.getoption("update_cache")


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def flm_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flm, rng)


@pytest.fixture
def flmn_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flmn, rng)


def s2fft_to_so3_sampling(s2fft_sampling: str) -> str:
    if s2fft_sampling.lower() == "mw":
        so3_sampling = "SO3_SAMPLING_MW"
    elif s2fft_sampling.lower() == "mwss":
        so3_sampling = "SO3_SAMPLING_MWSS"
    else:
        raise ValueError(
            f"Sampling scheme sampling={s2fft_sampling} not supported by so3."
        )

    return so3_sampling


def cache_subdirectory_path(cache_directory: Path, subdirectory: str) -> Path:
    cache_subdirectory = cache_directory / subdirectory
    if not cache_subdirectory.exists():
        cache_subdirectory.mkdir(parents=True)
    return cache_subdirectory


def cache_filename(parameters: dict, extension: str) -> str:
    return (
        "__".join(
            # Only recording floating-point values to 15 decimal significant digits to
            # avoid cache misses due to variations in less significant digits
            f"{k}={v:.15g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in parameters.items()
        )
        + "."
        + extension
    )


P = ParamSpec("P")
TestData: TypeAlias = Mapping[str, Any]


class TestDataFormat(NamedTuple):
    extension: str
    load: Callable[[Path], TestData]
    save: Callable[[Path, TestData], None]


def npz_load(path: Path) -> TestData:
    return np.load(path)


def npz_save(path: Path, data: TestData) -> None:
    return np.savez_compressed(path, **data)


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
    cache_directory: Path, use_cache: bool, update_cache: bool, seed: int
) -> Callable[[Callable[P, TestData], str], Callable[P, TestData]]:
    """Fixture (decorator) for loading/writing test data to/from the cache.

    Let `generate_data` be a function which generates test data.
    Applying this decorator to `generate_data` returns a function that takes the same
    arguments as `generate_data`, and which acts as:
    
    - If `use_cache` is `True`, attempt to load previously cached data from
      `cache_directory`.
      An error will be thrown if the cached data cannot be found.
    - Otherwise (`use_cache` is `False`), `generate_data` will be called to create the
      data to be used in the test.
    - If `update_cache` is `True`, any data generated for the test will be written to
      the cache (overwriting previous values if present). 

    Any generated test data will be written to a file named according to the keyword
    arguments passed to `generate_data`, under a module / function specific
    subdirectory in `cache_directory`.
    """

    def wrapper(
        generate_data: Callable[P, TestData], format: str
    ) -> Callable[P, TestData]:
        data_format = TEST_DATA_FORMATS[format]
        # Manually remove <> characters from <locals> instances to avoid filepath issues
        # on NTFS
        function_qualname = generate_data.__qualname__.replace("<", "").replace(">", "")
        cache_subdirectory = cache_subdirectory_path(
            cache_directory / generate_data.__module__, function_qualname
        )

        @wraps(generate_data)
        def cached_generate_data(*args: P.args, **kwargs: P.kwargs) -> TestData:
            call_args = inspect.getcallargs(generate_data, *args, **kwargs)
            cache_path = cache_subdirectory / cache_filename(
                {"seed": seed} | call_args, data_format.extension
            )
            if use_cache and not cache_path.exists():
                msg = f"Cache enabled but cached test data file {cache_path} not found."
                raise FileNotFoundError(msg)
            elif use_cache:
                data = data_format.load(cache_path)
            else:
                data = generate_data(*args, **kwargs)
            if update_cache:
                data_format.save(cache_path, data)
            return data

        return cached_generate_data

    return wrapper


@pytest.fixture
def cached_so3_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flmn_generator: Callable[..., np.ndarray],
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


@pytest.fixture
def cached_so3_samples_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
) -> Callable[P, TestData]:
    def generate_data(L: int, N: int, sampling: str) -> dict[str, np.ndarray]:
        import so3

        so3_parameters = so3.create_parameter_dict(
            L=L,
            N=N,
            sampling_scheme_str=s2fft_to_so3_sampling(sampling),
        )

        return {
            "f_size": so3.f_size(so3_parameters),
            "flmn_size": so3.flmn_size(so3_parameters),
            "n_alpha": so3.n_alpha(so3_parameters),
            "n_beta": so3.n_beta(so3_parameters),
            "n_gamma": so3.n_gamma(so3_parameters),
            "elmn2ind": {
                f"{el}_{m}_{n}": so3.elmn2ind(el, m, n, so3_parameters)
                for el in range(L)
                for m in range(-el, el + 1)
                for n in range(-N + 1, N)
            },
        }

    return cached_test_case_wrapper(generate_data, "json")


@pytest.fixture
def cached_ssht_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flm_generator: Callable[..., np.ndarray],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, L_lower: int, spin: int, sampling: str, reality: bool
    ) -> dict[str, np.ndarray]:
        import pyssht

        from s2fft.sampling import s2_samples

        flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
        f_ssht = pyssht.inverse(
            s2_samples.flm_2d_to_1d(flm, L),
            L,
            Method=sampling.upper(),
            Spin=spin,
            Reality=reality if spin == 0 else False,
        )
        return {"flm": flm, "f_ssht": f_ssht}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.fixture
def cached_healpy_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flm_generator: Callable[..., np.ndarray],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, nside: int, reality: bool, n_iter: int = 0
    ) -> dict[str, np.ndarray]:
        import healpy

        from s2fft.sampling import s2_samples

        flm = flm_generator(L=L, spin=0, reality=True)
        flm_hp = s2_samples.flm_2d_to_hp(flm, L)
        f_hp = healpy.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)
        flm_hp = healpy.sphtfunc.map2alm(f_hp, lmax=L - 1, iter=n_iter)
        return {"flm": flm, "f_hp": f_hp, "flm_hp": flm_hp}

    return cached_test_case_wrapper(generate_data, "npz")
