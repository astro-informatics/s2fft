import jax
import numpy as np
import pytest
import torch

from s2fft.utils import torch_wrapper


def sum_abs_square(x: jax.Array) -> float:
    return (abs(x) ** 2).sum()


@pytest.mark.parametrize("jax_function", [sum_abs_square])
@pytest.mark.parametrize("input_shape", [(), (1,), (2,), (3, 4)])
def test_wrap_as_torch_function_single_arg(rng, input_shape, jax_function):
    x_jax = jax.numpy.asarray(rng.standard_normal(input_shape))
    y_jax = jax_function(x_jax)
    x_torch = torch_wrapper.jax_array_to_torch_tensor(x_jax)
    assert isinstance(x_torch, torch.Tensor)
    torch_function = torch_wrapper.wrap_as_torch_function(jax_function)
    y_torch = torch_function(x_torch)
    assert isinstance(y_torch, torch.Tensor)
    np.testing.assert_allclose(np.asarray(y_jax), np.asarray(y_torch))
