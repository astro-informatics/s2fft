import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, ParamSpec, TypeAlias

import numpy as np
import pytest

P = ParamSpec("P")
TestData: TypeAlias = Mapping[str, Any]


class _TestDataFormat(NamedTuple):
    extension: str
    load: Callable[[Path], TestData]
    save: Callable[[Path, TestData], None]


def _npz_load(path: Path) -> TestData:
    return np.load(path)


def _npz_save(path: Path, data: TestData) -> None:
    return np.savez_compressed(path, **data)


def _json_load(path: Path) -> TestData:
    with path.open("r") as f:
        return json.load(f)


def _json_save(path: Path, data: TestData) -> None:
    with path.open("w") as f:
        json.dump(data, f)


@pytest.fixture(scope="session")
def _TEST_DATA_FORMATS() -> dict[str, _TestDataFormat]:
    """
    Lookup dict specifying methods to use when reading/writing data.

    Exposure as a constant (session-scoped) fixture allows use across the entire testing suite,
    without re-generation for each test. It is implicitly assumed that this fixture will not be
    mutated within other fixtures or the test cases themselves.
    """
    return {
        "npz": _TestDataFormat("npz", _npz_load, _npz_save),
        "json": _TestDataFormat("json", _json_load, _json_save),
    }
