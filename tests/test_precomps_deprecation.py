import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import s2fft
from s2fft.recursions import price_mcewen
from s2fft.sampling import s2_samples as samples
from s2fft.sampling import so3_samples
from s2fft.transforms import otf_recursions as otf

jax.config.update("jax_enable_x64", True)

L = 8
N = 2
PRECOMPS = [np.zeros(1)]


def _deprecations(record: list) -> list:
    """The warnings in `record` that s2fft raises for precomputed coefficients."""
    return [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning)
        and "evaluate their starting values and coefficients on the fly"
        in str(w.message)
    ]


def _calls(precomps: list | None) -> dict:
    """Calls of every public function that accepts precomps, with `precomps`."""
    f = np.zeros(samples.f_shape(L, "mw"))
    flm = np.zeros(samples.flm_shape(L), dtype=np.complex128)
    fw = np.zeros(so3_samples.f_shape(L, N, "mw"))
    flmn = np.zeros(so3_samples.flmn_shape(L, N), dtype=np.complex128)
    beta = samples.thetas(L, "mw")
    ftm = np.zeros(samples.ftm_shape(L, "mw"), dtype=np.complex128)
    return {
        "spherical forward numpy": lambda: s2fft.forward(f, L, precomps=precomps),
        "spherical forward jax": lambda: s2fft.forward(
            jnp.asarray(f), L, method="jax", precomps=precomps
        ),
        "spherical inverse numpy": lambda: s2fft.inverse(flm, L, precomps=precomps),
        "spherical inverse jax": lambda: s2fft.inverse(
            jnp.asarray(flm), L, method="jax", precomps=precomps
        ),
        "wigner forward numpy": lambda: s2fft.wigner.forward(
            fw, L, N, precomps=precomps
        ),
        "wigner forward jax": lambda: s2fft.wigner.forward(
            jnp.asarray(fw), L, N, method="jax", precomps=precomps
        ),
        "wigner inverse numpy": lambda: s2fft.wigner.inverse(
            flmn, L, N, precomps=precomps
        ),
        "wigner inverse jax": lambda: s2fft.wigner.inverse(
            jnp.asarray(flmn), L, N, method="jax", precomps=precomps
        ),
        "compute_all_slices": lambda: price_mcewen.compute_all_slices(
            beta, L, 1, precomps=precomps
        ),
        "compute_all_slices_jax": lambda: price_mcewen.compute_all_slices_jax(
            jnp.asarray(beta), L, 1, precomps=precomps
        ),
        "inverse_latitudinal_step": lambda: otf.inverse_latitudinal_step(
            flm, beta, L, 0, None, precomps=precomps
        ),
        "inverse_latitudinal_step_jax": lambda: otf.inverse_latitudinal_step_jax(
            jnp.asarray(flm), jnp.asarray(beta), L, 0, None, precomps=precomps
        ),
        "forward_latitudinal_step": lambda: otf.forward_latitudinal_step(
            ftm, beta, L, 0, None, precomps=precomps
        ),
        "forward_latitudinal_step_jax": lambda: otf.forward_latitudinal_step_jax(
            jnp.asarray(ftm), jnp.asarray(beta), L, 0, None, precomps=precomps
        ),
    }


@pytest.mark.parametrize("name", list(_calls(None)))
def test_precomps_argument_warns(name: str):
    # Exactly one warning for each call, attributed to the caller
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _calls(PRECOMPS)[name]()
    deprecations = _deprecations(record)
    assert len(deprecations) == 1
    assert "precomps argument is deprecated" in str(deprecations[0].message)
    assert deprecations[0].filename == __file__


@pytest.mark.parametrize("name", list(_calls(None)))
def test_no_warning_without_precomps(name: str):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _calls(None)[name]()
    assert not _deprecations(record)


@pytest.mark.parametrize(
    "generator, args",
    [
        (price_mcewen.generate_precomputes, (L,)),
        (price_mcewen.generate_precomputes_jax, (L,)),
        (price_mcewen.generate_precomputes_wigner, (L, N)),
        (price_mcewen.generate_precomputes_wigner_jax, (L, N)),
    ],
)
def test_generate_precomputes_warns(generator, args: tuple):
    # Once for each call, also for the Wigner generators, which call the others
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        generator(*args)
    deprecations = _deprecations(record)
    assert len(deprecations) == 1
    assert f"{generator.__name__} is deprecated" in str(deprecations[0].message)
    assert deprecations[0].filename == __file__


def test_precomps_warning_reaches_users():
    # With Python's default warning filters, as for a user's script
    script = (
        "import numpy as np\n"
        "import s2fft\n"
        "s2fft.forward(np.zeros((8, 15)), 8, precomps=[np.zeros(1)])\n"
    )
    result = subprocess.run(  # noqa: S603 (a fixed script)
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DeprecationWarning: The precomps argument is deprecated" in result.stderr
