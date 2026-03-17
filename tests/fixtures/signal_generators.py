from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from s2fft.utils import signal_generator


@pytest.fixture
def flm_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flm, rng)


@pytest.fixture
def flmn_generator(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    return partial(signal_generator.generate_flmn, rng)
