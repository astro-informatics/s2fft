from pathlib import Path

import pytest


@pytest.fixture
def cache_directory(request) -> Path:
    return request.config.getoption("cache_directory")


@pytest.fixture
def use_cache(request) -> Path:
    return request.config.getoption("use_cache")


@pytest.fixture
def update_cache(request) -> Path:
    return request.config.getoption("update_cache")
