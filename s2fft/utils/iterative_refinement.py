"""Iterative scheme for improving accuracy of linear transforms."""

from typing import Callable, TypeVar

T = TypeVar("T")


def forward_with_iterative_refinement(
    f: T,
    n_iter: int,
    forward_function: Callable[[T], T],
    backward_function: Callable[[T], T],
) -> T:
    """
    Apply forward transform with iterative refinement to improve accuracy.

    `Iterative refinement <https://en.wikipedia.org/wiki/Iterative_refinement>`_ is a
    general approach for improving the accuracy of numerial solutions to linear systems.
    In the context of spherical harmonic transforms, given a forward transform which is
    an _approximate_ inverse to a corresponding backward ('inverse') transform,
    iterative refinement allows defining an iterative forward transform which is a more
    accurate

    Args:
        f: Array argument to forward transform (signal on sphere) to compute iteratively
           refined forward transform at.

        n_iter: Number of refinement iterations to use, non-negative.

        forward_function: Function computing forward transform (approximate inverse of
            backward transform).

        backward_function: Function computing backward ('inverse') transform.

    Returns:
        Array output from iteratively refined forward transform (spherical harmonic
        coefficients).

    """
    flm = forward_function(f)
    for _ in range(n_iter):
        f_recov = backward_function(flm)
        f_error = f - f_recov
        flm += forward_function(f_error)
    return flm
