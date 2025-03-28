from s2fft.utils import quadrature_jax as _quadrature_jax
from s2fft.utils import torch_wrapper as _torch_wrapper

_torch_wrapper.populate_namespace_by_wrapping_functions_in_module(
    globals(), _quadrature_jax
)
