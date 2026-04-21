import jax.numpy as jnp
import numpy as np


def compatible_complex_dtype(f: jnp.ndarray | np.ndarray) -> str:
    """
    Return the (string specifier of the) smallest complex dtype compatible with ``f``.

    The smallest complex dtype that is compatible with ``f`` is the complex dtype that
    does not loose precision when casting the values in ``f`` to complex numbers. If
    ``f`` is already a complex array, ``f``'s data-type is simply returned. Otherwise,
    ``f`` must be of ``float{XX}`` dtype, in which case ``complex{2XX}`` is returned.

    Notes:
        At present, this is just a wrapper around a lookup function. Only input arrays
        with types ``complex128``, ``complex64``, ``float64``, or ``float32`` are supported.

    """
    conversions = {
        "complex128": "complex128",
        "float64": "complex128",
        "complex64": "complex64",
        "float32": "complex64",
    }
    torch_conversions = {
        f"torch.{key}": f"torch.{value}" for key, value in conversions.items()
    }
    dtype = str(f.dtype)
    return {**conversions, **torch_conversions}[dtype]
