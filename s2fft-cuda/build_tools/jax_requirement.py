"""Dynamic metadata for the s2fft CUDA plugins (scikit-build-core local provider).

The plugins are compiled against the XLA FFI headers of the jax installed in the build
environment (``jax.ffi.include_dir()``, s2fft-cuda/CMakeLists.txt). Their handlers do not
register with an older jax (tested: built against jax 0.12, they fail on jax 0.7.2 and
0.8.0), so the plugin requires at least the jax it was built against.
"""


def dynamic_metadata(field, settings, project):  # noqa: ARG001
    if field != "dependencies":
        msg = f"jax_requirement only provides dependencies, not {field}"
        raise RuntimeError(msg)
    import jax

    return [f"jax>={jax.__version__}"]
