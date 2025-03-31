#  conda activate s2fft_env

import numpy as np
import pytest
from jax import config


from s2fft.base_transforms import spherical
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch

