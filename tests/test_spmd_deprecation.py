import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import s2fft
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import otf_recursions as otf
from s2fft.transforms import spherical

jax.config.update("jax_enable_x64", True)

L = 8


def _spmd_deprecations(record: list) -> list:
    return [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning)
        and "spmd argument is deprecated" in str(w.message)
    ]


def _calls() -> dict:
    """Calls of each public JAX function with the spmd argument, given spmd."""
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal(samples.f_shape(L, "mw")))
    flm = jnp.asarray(s2fft.forward(np.asarray(f), L, method="numpy"))
    beta = jnp.asarray(samples.thetas(L, "mw"))
    ftm = jnp.asarray(
        rng.standard_normal(samples.ftm_shape(L, "mw")) + 0j, dtype=jnp.complex128
    )
    return {
        "forward": lambda spmd: s2fft.forward(f, L, method="jax", spmd=spmd),
        "inverse": lambda spmd: s2fft.inverse(flm, L, method="jax", spmd=spmd),
        "forward_jax": lambda spmd: spherical.forward_jax(f, L, spmd=spmd),
        "inverse_jax": lambda spmd: spherical.inverse_jax(flm, L, spmd=spmd),
        "forward_latitudinal_step_jax": lambda spmd: otf.forward_latitudinal_step_jax(
            ftm, beta, L, 0, None, spmd=spmd
        ),
        "inverse_latitudinal_step_jax": lambda spmd: otf.inverse_latitudinal_step_jax(
            flm, beta, L, 0, None, spmd=spmd
        ),
    }


@pytest.mark.parametrize("name", list(_calls()))
def test_spmd_warns_and_runs_on_one_device(name: str):
    # Each call warns once, attributed to the caller, also when repeated: the jitted
    # transforms warn from unjitted wrappers, not while they are traced
    call = _calls()[name]
    for _ in range(2):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = call(True)
        deprecations = _spmd_deprecations(record)
        assert len(deprecations) == 1
        assert deprecations[0].filename == __file__
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        expected = call(False)
    assert not _spmd_deprecations(record)
    np.testing.assert_array_equal(result, expected)


def test_spmd_warning_reaches_users():
    # With Python's default warning filters, as for a user's script: once for each
    # line, so for both calls of each jitted transform
    script = (
        "import numpy as np\n"
        "import s2fft\n"
        "f, flm = np.zeros((8, 15)), np.zeros((8, 15), dtype=complex)\n"
        "s2fft.forward_jax(f, 8, spmd=True)\n"
        "s2fft.forward_jax(f, 8, spmd=True)\n"
        "s2fft.inverse_jax(flm, 8, spmd=True)\n"
        "s2fft.inverse_jax(flm, 8, spmd=True)\n"
    )
    result = subprocess.run(  # noqa: S603 (a fixed script)
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        result.stderr.count("DeprecationWarning: The spmd argument is deprecated") == 4
    )
