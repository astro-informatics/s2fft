"""
Dynamic metadata for the s2fft CUDA plugins (scikit-build-core local provider).

The plugins are compiled against the XLA FFI headers of the jax installed in the build
environment (``jax.ffi.include_dir()``, s2fft-cuda/CMakeLists.txt). Their handlers do not
register with an older jax (tested: built against jax 0.12, they fail on jax 0.7.2 and
0.8.0), so the plugin requires at least the jax it was built against.
"""


def dynamic_metadata(field, settings, project):  # noqa: ARG001
    """
    Return the dependencies of the plugin: the jax it is built against, or newer.

    Called by scikit-build-core for ``metadata.dependencies.provider = "jax_requirement"``.

    Args:
        field (str): Metadata field to provide; only ``"dependencies"`` is supported.

        settings (dict): Provider settings from ``pyproject.toml`` (unused).

        project (dict): The ``[project]`` table of ``pyproject.toml`` (unused).

    Raises:
        RuntimeError: If a field other than ``"dependencies"`` is requested.

    Returns:
        list[str]: The requirement ``jax>=<version of the jax in the build environment>``.

    """
    if field != "dependencies":
        msg = f"jax_requirement only provides dependencies, not {field}"
        raise RuntimeError(msg)
    import jax

    return [f"jax>={jax.__version__}"]
