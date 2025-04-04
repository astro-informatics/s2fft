import jax
import numpy as np
import pytest
import torch
from jax.test_util import check_grads
from jax.tree_util import tree_all, tree_map

from s2fft.utils import torch_wrapper
from s2fft.utils.signal_generator import complex_normal

jax.config.update("jax_enable_x64", True)


def sum_abs_square(x: jax.Array) -> float:
    return (abs(x) ** 2).sum()


def log_sum_exp(x: jax.Array) -> float:
    max_x = x.max()
    return max_x + jax.numpy.log(jax.numpy.exp(x - max_x).sum())


def cubic(x: jax.Array) -> jax.Array:
    return x**3 - 2 * x**2 + 3 * x - 1


def conj(x: jax.Array) -> jax.Array:
    return x.conj()


DTYPES = ["float32", "float64", "complex64", "complex128"]

INPUT_SHAPES = [(), (1,), (2,), (3, 4)]

PYTREE_STRUCTURES = [(), [(), ((1,), (2, 3))], {"a": [(1,), ()], "b": {"0": (1, 2)}}]

JAX_SINGLE_ARG_FUNCTIONS = [sum_abs_square, log_sum_exp, cubic, conj]


def generate_standard_normal(rng, shape, dtype):
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape, dtype=dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        return complex_normal(rng, shape, dtype=dtype)
    else:
        msg = f"dtype {dtype} must be a floating or complex floating data type"
        raise ValueError(msg)


def generate_pytree(rng, converter, dtype, structure):
    if isinstance(structure, tuple):
        if structure == () or all(isinstance(child, int) for child in structure):
            return converter(generate_standard_normal(rng, structure, dtype))
        else:
            return tuple(
                generate_pytree(rng, converter, dtype, child) for child in structure
            )
    elif isinstance(structure, list):
        return [generate_pytree(rng, converter, dtype, child) for child in structure]
    elif isinstance(structure, dict):
        return {
            key: generate_pytree(rng, converter, dtype, value)
            for key, value in structure.items()
        }
    else:
        raise TypeError(
            f"pytree structure with type {type(structure)} not of recognised type"
        )


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_jax_array_to_torch_tensor(rng, input_shape, dtype):
    x_jax = jax.numpy.asarray(generate_standard_normal(rng, input_shape, dtype=dtype))
    x_torch = torch_wrapper.jax_array_to_torch_tensor(x_jax)
    assert isinstance(x_torch, torch.Tensor)
    assert x_torch.dtype == getattr(torch, dtype)
    np.testing.assert_allclose(np.asarray(x_jax), np.asarray(x_torch))


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_tensor_to_jax_array(rng, input_shape, dtype):
    x_torch = torch.from_numpy(generate_standard_normal(rng, input_shape, dtype=dtype))
    x_jax = torch_wrapper.torch_tensor_to_jax_array(x_torch)
    assert isinstance(x_jax, jax.Array)
    assert x_jax.dtype == dtype
    np.testing.assert_allclose(np.asarray(x_jax), np.asarray(x_torch))


@pytest.mark.parametrize("pytree_structure", PYTREE_STRUCTURES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tree_map_jax_array_to_torch_tensor(rng, pytree_structure, dtype):
    jax_pytree = generate_pytree(rng, jax.numpy.asarray, dtype, pytree_structure)
    torch_pytree = torch_wrapper.tree_map_jax_array_to_torch_tensor(jax_pytree)
    assert tree_all(
        tree_map(lambda leaf: isinstance(leaf, jax.Array), jax_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: leaf.dtype == dtype, jax_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: isinstance(leaf, torch.Tensor), torch_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: leaf.dtype == getattr(torch, dtype), torch_pytree),
    )
    assert tree_all(
        tree_map(
            lambda leaf_1, leaf_2: np.allclose(np.asarray(leaf_1), np.asarray(leaf_2)),
            torch_pytree,
            jax_pytree,
        )
    )


@pytest.mark.parametrize("pytree_structure", PYTREE_STRUCTURES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tree_map_torch_tensor_to_jax_array(rng, pytree_structure, dtype):
    torch_pytree = generate_pytree(rng, torch.from_numpy, dtype, pytree_structure)
    jax_pytree = torch_wrapper.tree_map_torch_tensor_to_jax_array(torch_pytree)
    assert tree_all(
        tree_map(lambda leaf: isinstance(leaf, jax.Array), jax_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: leaf.dtype == dtype, jax_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: isinstance(leaf, torch.Tensor), torch_pytree),
    )
    assert tree_all(
        tree_map(lambda leaf: leaf.dtype == getattr(torch, dtype), torch_pytree),
    )
    assert tree_all(
        tree_map(
            lambda leaf_1, leaf_2: np.allclose(np.asarray(leaf_1), np.asarray(leaf_2)),
            torch_pytree,
            jax_pytree,
        )
    )


@pytest.mark.parametrize("jax_function", JAX_SINGLE_ARG_FUNCTIONS)
@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_wrap_as_torch_function_single_arg(rng, input_shape, dtype, jax_function):
    x_numpy = generate_standard_normal(rng, input_shape, dtype=dtype)
    x_jax = jax.numpy.asarray(x_numpy)
    y_jax, vjp_jax_function = jax.vjp(jax_function, x_jax)
    x_torch = torch.tensor(x_numpy, requires_grad=True)
    torch_function = torch_wrapper.wrap_as_torch_function(jax_function)
    y_torch = torch_function(x_torch)
    assert isinstance(y_torch, torch.Tensor)
    y_dtype = str(y_jax.dtype)
    assert y_torch.dtype == getattr(torch, y_dtype)
    np.testing.assert_allclose(np.asarray(y_jax), np.asarray(y_torch.detach()))
    y_bar = generate_standard_normal(rng, y_jax.shape, dtype=y_dtype)
    # JAX and PyTorch use different conventions for derivatives of complex functions
    # (see https://github.com/jax-ml/jax/issues/4891) so we need to conjugate the
    # inputs to and outputs from VJP to get equivalent behaviour to backward method on
    # torch tensors
    x_bar_jax = vjp_jax_function(y_bar.conj())[0].conj()
    y_torch.backward(torch.from_numpy(y_bar))
    assert x_torch.grad.dtype == getattr(torch, dtype)
    np.testing.assert_allclose(
        np.asarray(x_bar_jax), np.asarray(x_torch.grad.resolve_conj())
    )


# torch.autograd.gradcheck tolerances calibrated for double precision so only do checks
# with double precision floating point types
@pytest.mark.parametrize("jax_function", JAX_SINGLE_ARG_FUNCTIONS)
@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_wrap_as_torch_function_single_arg_autograd_check(
    rng, input_shape, dtype, jax_function
):
    x_numpy = generate_standard_normal(rng, input_shape, dtype=dtype)
    x_jax = jax.numpy.asarray(x_numpy)
    check_grads(jax_function, (x_jax,), order=1)
    x_torch = torch.tensor(x_numpy, requires_grad=True)
    torch_function = torch_wrapper.wrap_as_torch_function(jax_function)
    torch.autograd.gradcheck(torch_function, x_torch)
