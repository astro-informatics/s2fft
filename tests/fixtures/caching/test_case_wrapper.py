import inspect
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeAlias

import numpy as np
import pytest


def _cache_subdirectory_path(cache_directory: Path, subdirectory: str) -> Path:
    cache_subdirectory = cache_directory / subdirectory
    if not cache_subdirectory.exists():
        cache_subdirectory.mkdir(parents=True)
    return cache_subdirectory


def _cache_filename(parameters: dict, extension: str) -> str:
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


@pytest.fixture(scope="session")
def _s2fft_to_so3_sampling() -> Callable[[str], str]:
    """Internal conversion function from s2fft scheme specifiers to SO3 scheme specifiers."""

    def _inner(s2fft_sampling: str) -> str:
        if s2fft_sampling.lower() == "mw":
            so3_sampling = "SO3_SAMPLING_MW"
        elif s2fft_sampling.lower() == "mwss":
            so3_sampling = "SO3_SAMPLING_MWSS"
        else:
            raise ValueError(
                f"Sampling scheme sampling={s2fft_sampling} not supported by so3."
            )

        return so3_sampling

    return _inner


P = ParamSpec("P")
TestData: TypeAlias = Mapping[str, Any]


@pytest.fixture
def cached_test_case_wrapper(
    cache_directory: Path,
    use_cache: bool,
    update_cache: bool,
    seed: int,
    _TEST_DATA_FORMATS: dict[str, TestData],
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
        data_format = _TEST_DATA_FORMATS[format]
        # Manually remove <> characters from <locals> instances to avoid filepath issues
        # on NTFS
        function_qualname = generate_data.__qualname__.replace("<", "").replace(">", "")
        cache_subdirectory = _cache_subdirectory_path(
            cache_directory / generate_data.__module__, function_qualname
        )

        @wraps(generate_data)
        def cached_generate_data(*args: P.args, **kwargs: P.kwargs) -> TestData:
            call_args = inspect.getcallargs(generate_data, *args, **kwargs)
            cache_path = cache_subdirectory / _cache_filename(
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
    _s2fft_to_so3_sampling: Callable[[str], str],
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
            sampling_scheme_str=_s2fft_to_so3_sampling(sampling),
            reality=False,
        )

        f_so3 = so3.inverse(so3_samples.flmn_3d_to_1d(flmn, L, N), so3_parameters)
        flmn_so3 = so3_samples.flmn_1d_to_3d(so3.forward(f_so3, so3_parameters), L, N)

        return {"flmn": flmn, "f_so3": f_so3, "flmn_so3": flmn_so3}

    return cached_test_case_wrapper(generate_data, "npz")
