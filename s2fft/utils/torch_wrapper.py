"""
Utilities for wrapping JAX functions for use in PyTorch.

Based on Gist by Matt Johnson at
https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9

and jax2torch package by Phil Wang
https://github.com/lucidrains/jax2torch

which is released under a MIT license

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from functools import wraps
from inspect import getmembers, isroutine, signature
from types import ModuleType
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import jax
import jax.dlpack
from jax.tree_util import tree_map

try:
    import torch
    import torch.utils.dlpack

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

T = TypeVar("T")
PyTree = Union[Dict[Any, "PyTree"], List["PyTree"], Tuple["PyTree"], T]


def check_torch_available() -> None:
    """Raise an error if Torch is not importable."""
    if not TORCH_AVAILABLE:
        msg = (
            "torch needs to be installed to use torch wrapper functionality but could\n"
            "not be imported. Install s2fft with torch extra using:\n"
            "    pip install s2fft[torch]\n"
            "to allow use of torch wrapper functionality."
        )
        raise RuntimeError(msg)


def jax_array_to_torch_tensor(jax_array: jax.Array) -> torch.Tensor:
    """
    Convert from JAX array to Torch tensor via mutual DLPack support.

    Args:
       jax_array: JAX array to convert.

    Returns:
       Torch tensor object with equivalent data to `jax_array`.

    """
    return torch.utils.dlpack.from_dlpack(jax_array)


def torch_tensor_to_jax_array(torch_tensor: torch.Tensor) -> jax.Array:
    """
    Convert from Torch tensor to JAX array via mutual DLPack support.

    Args:
       torch_tensor: Torch tensor to convert.

    Returns:
       JAX array object with equivalent data to `torch_tensor`.

    """
    # JAX currently only support DLPack arrays with trivial strides so force torch
    # tensor to be contiguous before DLPack conversion
    # https://github.com/google/jax/issues/8082
    torch_tensor = torch_tensor.contiguous()
    # Torch does lazy conjugation using flag bits and DLPack does not support this
    # https://github.com/data-apis/array-api-compat/issues/173#issuecomment-2272192054
    # so we explicitly resolve any conjugacy operations implied by bit before conversion
    torch_tensor = torch_tensor.resolve_conj()
    # DLPack compatibility does support tensors that require gradient so detach. As
    # this intended for use when wrapping JAX code detaching tensor from gradient values
    # should not be problematic as derivatives will be separately routed via JAX
    torch_tensor = torch_tensor.detach()
    return jax.dlpack.from_dlpack(torch_tensor)


def tree_map_jax_array_to_torch_tensor(
    jax_pytree: PyTree[jax.Array],
) -> PyTree[torch.Tensor]:
    """
    Convert from a pytree with JAX arrays to corresponding pytree with Torch tensors.

    Args:
       jax_pytree: Pytree of JAX arrays or non-array values.

    Returns:
       Pytree with equivalent structure but any JAX arrays mapped to Torch tensors.

    """
    return tree_map(
        lambda t: jax_array_to_torch_tensor(t) if isinstance(t, jax.Array) else t,
        jax_pytree,
    )


def tree_map_torch_tensor_to_jax_array(
    torch_pytree: PyTree[torch.Tensor],
) -> PyTree[jax.Array]:
    """
    Convert from a pytree with Torch tensors to corresponding pytree with JAX arrays.

    Args:
       torch_pytree: Pytree of Torch tensorss or non-array values.

    Returns:
       Pytree with equivalent structure but any Torch tensors mapped to JAX arrays.

    """
    return tree_map(
        lambda t: torch_tensor_to_jax_array(t) if isinstance(t, torch.Tensor) else t,
        torch_pytree,
    )


def wrap_as_torch_function(
    jax_function: Callable, differentiable_argnames: None | tuple[str] = None
) -> Callable:
    """
    Wrap a function implemented using JAX API to be callable within Torch.

    Deals with conversion of argument(s) from JAX array(s) to Torch tensor(s), and of
    return value(s) from JAX array(s) to Torch tensor(s), as well as recording
    context needed to compute reverse-mode gradients in Torch using JAX automatic
    differentiation support if differentiable arguments are present.

    Args:
        jax_function: JAX function to wrap.
        differentiable_argnames: Names of arguments of `jax_function` which function
           output(s) are differentiable with respect to, and gradients should be
           compute with respect to in Torch backwards pass. If `None` (the default)
           the names of all arguments which are annotated as being `jax.Array` instances
           will be used.

    Returns:
        Wrapped function callable from Torch.

    """
    check_torch_available()
    sig = signature(jax_function)
    if differentiable_argnames is None:
        differentiable_argnames = tuple(
            name
            for name, param in sig.parameters.items()
            if issubclass(param.annotation, jax.Array)
        )
    for argname in differentiable_argnames:
        if argname not in sig.parameters:
            msg = f"{argname} passed is not a valid argument to {jax_function}"
            raise ValueError(msg)

    @wraps(jax_function)
    def torch_function(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        differentiable_args = tuple(
            bound_args.arguments[argname] for argname in differentiable_argnames
        )

        def jax_function_diff_args_only(*differentiable_args):
            for key, value in zip(differentiable_argnames, differentiable_args):
                bound_args.arguments[key] = value
            return jax_function(*bound_args.args, **bound_args.kwargs)

        class WrappedJaxFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map_torch_tensor_to_jax_array(args)
                primals_out, ctx.vjp = jax.vjp(jax_function_diff_args_only, *args)
                return tree_map_jax_array_to_torch_tensor(primals_out)

            @staticmethod
            def backward(ctx, *grad_outputs):
                # JAX and PyTorch use different conventions for derivatives of complex
                # functions (see https://github.com/jax-ml/jax/issues/4891) so we need
                # to conjugate the inputs to and outputs from VJP to get equivalent
                # behaviour to backward method on torch tensors
                grad_outputs = tree_map(lambda g: g.conj(), grad_outputs)
                jax_grad_outputs = tree_map_torch_tensor_to_jax_array(grad_outputs)
                jax_grad_inputs = ctx.vjp(*jax_grad_outputs)
                grad_inputs = tree_map_jax_array_to_torch_tensor(jax_grad_inputs)
                return tree_map(lambda g: g.conj(), grad_inputs)

        return WrappedJaxFunction.apply(*differentiable_args)

    docstring_replacements = {
        "JAX": "Torch",
        "jnp.ndarray": "torch.Tensor",
        "jax.Array": "torch.Tensor",
    }
    if torch_function.__doc__ is not None:
        for original, new in docstring_replacements.items():
            torch_function.__doc__ = torch_function.__doc__.replace(original, new)

    torch_function.__annotations__ = torch_function.__annotations__.copy()
    for name, annotation in torch_function.__annotations__.items():
        if isinstance(annotation, type) and issubclass(annotation, jax.Array):
            torch_function.__annotations__[name] = torch.Tensor

    return torch_function


def populate_namespace_by_wrapping_functions_in_module(
    namespace: dict, module: ModuleType
) -> None:
    """
    Populate a namespace by wrapping all (JAX) functions in a module as Torch functions.

    Args:
        namespace: Namespace to define wrapped functions in.
        module: Source module for (JAX) functions to wrap. Note all functions in this
            module without a preceding underscore in their name will be wrapped
            irrespective of whether they are defined in the module or not.

    """
    for name, function in getmembers(module, isroutine):
        if not name.startswith("_"):
            namespace[name] = wrap_as_torch_function(function)
