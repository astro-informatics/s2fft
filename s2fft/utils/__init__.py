from . import quadrature
from . import quadrature_jax
from . import quadrature_torch
from . import resampling
from . import resampling_jax
from . import resampling_torch
from . import healpix_ffts
from . import signal_generator
from . import rotation
from . import jax_pritimive

from jax.lib import xla_client
from s2fft_lib import _s2fft

for name, fn in _s2fft.registration().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")
