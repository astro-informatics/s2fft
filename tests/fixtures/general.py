import numpy as np
import pytest


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
