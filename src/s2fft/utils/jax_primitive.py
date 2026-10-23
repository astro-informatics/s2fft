from collections.abc import Callable
from functools import partial

from jax.extend import core
from jax.interpreters import ad, batching, mlir, xla


def register_primitive(
    name: str,
    multiple_results: bool,
    abstract_evaluation: Callable,
    lowering_per_platform: dict[None | str, Callable],
    batcher: Callable | None = None,
    jacobian_vector_product: Callable | None = None,
    transpose: Callable | None = None,
    is_linear: bool = False,
):
    """
    Register a new custom JAX primitive.

    This function provides a streamlined way to register custom JAX primitives,
    including their implementation, abstract evaluation, lowering rules for different
    platforms, and optional rules for batching and automatic differentiation.

    Args:
        name (str): The name of the primitive.
        multiple_results (bool): A boolean indicating whether the primitive returns multiple values.
        abstract_evaluation (Callable): A callable that defines the abstract evaluation rule for the primitive.
            It should take `ShapedArray` instances as inputs and return `ShapedArray` instances for the outputs.
        lowering_per_platform (Dict[Union[None, str], Callable]): A dictionary mapping platform names
            (e.g., "cpu", "gpu", or None for platform-independent) to their respective lowering rules.
            A lowering rule translates the primitive into a sequence of MLIR operations.
        batcher (Optional[Callable]): An optional callable that defines the batched evaluation rule for the primitive.
            This is used by JAX's automatic batching (vmap).
        jacobian_vector_product (Optional[Callable]): An optional callable that defines the Jacobian-vector product
            (JVP) rule for the primitive. This is used for forward-mode automatic differentiation.
        transpose (Optional[Callable]): An optional callable that defines the transpose rule for the primitive.
            This is used for reverse-mode automatic differentiation (autograd).
        is_linear (bool): A boolean indicating whether the primitive is linear. If True and a `transpose` rule
            is provided, `ad.deflinear` is used, which can optimize linear operations.

    Returns:
        jax.core.Primitive: The registered custom JAX primitive object.

    Raises:
        ValueError: If an invalid platform is specified in `lowering_per_platform`.

    """
    # Step 1: Create a new JAX primitive with the given name.
    primitive = core.Primitive(name)

    # Step 2: Set the `multiple_results` attribute of the primitive.
    primitive.multiple_results = multiple_results

    # Step 3: Define the default implementation of the primitive using `xla.apply_primitive`.
    # This means that by default, the primitive will be lowered to XLA.
    primitive.def_impl(partial(xla.apply_primitive, primitive))

    # Step 4: Register the abstract evaluation rule for the primitive.
    # This rule tells JAX how to infer the shape and dtype of the primitive's outputs
    # given its inputs, without actually executing the computation.
    primitive.def_abstract_eval(abstract_evaluation)

    # Step 5: Register lowering rules for the primitive across different platforms.
    # This step defines how the primitive is translated into lower-level operations
    # (e.g., MLIR) for execution on specific hardware (CPU, GPU, etc.).
    for platform, lowering in lowering_per_platform.items():
        mlir.register_lowering(primitive, lowering, platform=platform)

    # Step 6: Register the batching rule if provided.
    # The batching rule enables JAX's `vmap` transformation to work with this primitive.
    if batcher is not None:
        batching.primitive_batchers[primitive] = batcher

    # Step 7: Register the Jacobian-vector product (JVP) rule if provided.
    # The JVP rule is essential for forward-mode automatic differentiation.
    if jacobian_vector_product is not None:
        ad.primitive_jvps[primitive] = jacobian_vector_product

    # Step 8: Register the transpose rule if provided.
    # The transpose rule is crucial for reverse-mode automatic differentiation (autograd).
    if transpose is not None:
        if is_linear:
            # If the primitive is linear, use `ad.deflinear` for optimized transpose registration.
            ad.deflinear(primitive, transpose)
        else:
            # Otherwise, use `ad.primitive_transposes` for general transpose registration.
            ad.primitive_transposes[primitive] = transpose

    # Step 9: Return the newly registered primitive.
    return primitive
