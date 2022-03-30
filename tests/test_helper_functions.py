import pytest
from pytest import fixture
import numpy as np
import pyssht as ssht
import s2fft as s2f


@fixture
def ssht_signal_generator():
    return s2f.utils.generate_signal_ssht
