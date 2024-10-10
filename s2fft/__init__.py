import logging

import jax

from . import logs
from .recursions.price_mcewen import (
    generate_precomputes,
    generate_precomputes_jax,
    generate_precomputes_wigner,
    generate_precomputes_wigner_jax,
)
from .transforms import wigner
from .transforms.spherical import (
    forward,
    forward_jax,
    forward_numpy,
    inverse,
    inverse_jax,
    inverse_numpy,
)
from .utils.rotation import generate_rotate_dls, rotate_flms

if jax.config.read("jax_enable_x64") is False:
    logger = logging.getLogger("s2fft")
    logger.warning(
        "JAX is not using 64-bit precision. This will dramatically affect numerical precision at even moderate L."
    )
