from s2fft.utils import resampling_jax as _resampling_jax
from s2fft.utils import torch_wrapper as _torch_wrapper

_torch_wrapper.populate_namespace_by_wrapping_functions_in_module(
    globals(), _resampling_jax
)
