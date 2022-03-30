import pytest
from pytest import fixture
import numpy as np
import s2fft as s2f


@fixture
def flm_generator():
    return s2f.utils.generate_flm
