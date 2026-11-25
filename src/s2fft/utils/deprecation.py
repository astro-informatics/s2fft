"""Deprecation warnings for the precomputed Wigner-d recursion coefficients."""

import warnings


def warn_precomps_deprecated(function: str | None = None, stacklevel: int = 2) -> None:
    """
    Warn that precomputed Wigner-d recursion coefficients are deprecated.

    The recursions evaluate their starting values and coefficients on the fly, so
    s2fft no longer uses precomputed ones: the ``precomps`` arguments of the transforms
    are ignored, and the functions that generate them are kept only for backward
    compatibility. Both will be removed in a future release.

    Args:
        function (str, optional): Name of the deprecated function that generates the
            coefficients, or None for a ``precomps`` argument. Defaults to None.

        stacklevel (int, optional): As for :func:`warnings.warn`, counted from the
            caller of this function: the default 2 points the warning at the code that
            called it. Defaults to 2.

    """
    subject = (
        "The precomps argument is deprecated and ignored"
        if function is None
        else f"{function} is deprecated, since s2fft no longer uses its output"
    )
    warnings.warn(
        f"{subject}: the Wigner-d recursions evaluate their starting values and "
        "coefficients on the fly. It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )
