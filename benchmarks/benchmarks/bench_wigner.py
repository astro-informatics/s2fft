import numpy as np

from functools import partial
import timeit as timeit
import importlib

import jax.numpy as jnp
import jax as jax
from jax.config import config

config.update("jax_enable_x64", True)  # this only works on startup!

import s2fft.wigner as wigner
import s2fft.samples as samples
import pyssht as ssht
