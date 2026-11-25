import json
import os
import re
import subprocess
import sys

import jax
import numpy as np
import pytest
import torch

import s2fft
from s2fft.precompute_transforms.spherical import forward, inverse
from s2fft.sampling import s2_samples as samples
from s2fft.transforms import spherical, wigner
from s2fft.utils import signal_generator
from s2fft.utils._dtype_association import compatible_complex_dtype

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix", "cc", "f2"])
@pytest.mark.parametrize("fwd", [True, False], ids=("forward", "inverse"))
@pytest.mark.parametrize(
    "downsample_kernel", [True, False], ids=("downsample kernel", "full kernel")
)
@pytest.mark.parametrize("method", ["jax", "torch"])
@pytest.mark.parametrize(
    "reality", [True, False], ids=("real signal", "complex signal")
)
def test_lower_precision_transforms(
    get_flm_and_precompute_kernel,
    fwd: bool,
    downsample_kernel: bool,
    sampling: str,
    method: str,
    reality: bool,
    L: int = 64,
    spin: int = 0,
    recursion: str = "auto",
    log_ratio_range: tuple[float, float] = (1.5, 3.0),
) -> None:
    """
    Verify that signal arrays inherit the dtype of the input harmonic coefficient arrays.

    Test is run as a matrix across:
    - fwd (to test both the forward [True] and inverse [False] transforms)
    - downsample_kernel (to test that the kernel dtype is ignored when deciding on the output dtype)
    - sampling (to ensure there are no code-paths that are still forcing array creation with a fixed dtype)
    - reality (effectively handles the two cases for real signals and complex signals). Note that for the inverse transform, the output is returned as a floatXX array if reality is set to True, however this conversion doesn't
    actually occur until after all computations have been conducted.
    - method (ensure that dtype behaviour occurs for all of the jax / torch paths)

    Verification takes two forms:
    - A confirmation that the output array has the expected dtype (based on the signal dtype input)
    - A check on the orders of magnitude of the round trip error for both the long-precision and short-precision computations.
      Specifically, it is checked that the OoM for the short precision calculation is half that of the long-precision
      calculation, plus or minus 1 (to allow for rounding and near-misses).
    """
    nside = L // 2 if sampling == "healpix" else None
    common_args = {
        "L": L,
        "spin": spin,
        "nside": nside,
        "sampling": sampling,
        "reality": reality,
        "method": method,
    }

    # Generate flm coefficients and signal to use as "baseline truths"
    flm, kernel = get_flm_and_precompute_kernel(
        recursion=recursion,
        forward=fwd,
        **common_args,
    )
    f = s2fft.inverse(flm, **common_args)

    # Establish;
    # - which array is being transformed
    # - which transform method to call
    # - the 'true values' that should be recovered
    # - the down-cast input array dtype & the lower-precision input
    # - the expected dtype of the lower-precision calculation output
    casting_method = "astype" if method != "torch" else "to"
    if fwd:
        to_transform = f
        transform_direction = forward
        true_values = flm

        single_dtype = "float32" if reality else "complex64"
        if method == "torch":
            single_dtype = getattr(torch, single_dtype)

        to_transform_lower_precision = getattr(to_transform, casting_method)(
            single_dtype
        )
        # forward transform should result in complex array output,
        # even if the signal is real.
        expected_single_dtype = compatible_complex_dtype(to_transform_lower_precision)
    else:
        to_transform = flm
        transform_direction = inverse
        true_values = f

        single_dtype = "complex64"
        # Inverse transform should cast to real arrays if reality is specified.
        expected_single_dtype = "float32" if reality else "complex64"
        if method == "torch":
            single_dtype = getattr(torch, single_dtype)
            expected_single_dtype = getattr(torch, expected_single_dtype)
            to_transform = torch.as_tensor(to_transform)

        to_transform_lower_precision = getattr(to_transform, casting_method)(
            single_dtype
        )

    # Torch things to avoid operation errors when comparing to numpy arrays
    if method == "torch":
        true_values = torch.as_tensor(true_values)

    # Down-cast the kernel if instructed to do so
    kernel_to_use = (
        getattr(kernel, casting_method)(single_dtype) if downsample_kernel else kernel
    )

    # Perform transform in double precision
    double_calc_result = transform_direction(
        to_transform,
        kernel=kernel,
        **common_args,
    )
    # Perform transform in single precision, possibly with down-cast kernel
    single_calc_result = transform_direction(
        to_transform_lower_precision,
        kernel=kernel_to_use,
        **common_args,
    )

    # Confirm that the output inherits the lower precision dtype
    short_result_dtype = single_calc_result.dtype
    assert str(short_result_dtype) == str(expected_single_dtype)

    # Check expectations for the error. 1/2 precision ~= 1/2 the error order of magnitude (OOM).
    # HEALPix lacks a sampling theorem though, so we don't check this holds in that case.
    no_sampling_theorem = sampling == "healpix" and fwd
    if not no_sampling_theorem:
        log_round_trip_error_double = np.log10(
            abs(true_values - double_calc_result).max()
        )
        log_round_trip_error_single = np.log10(
            abs(true_values - single_calc_result).max()
        )
        log_error_ratio = log_round_trip_error_double / log_round_trip_error_single

        assert log_ratio_range[0] <= log_error_ratio <= log_ratio_range[1]


# Run by test_otf_transforms_x64_disabled in subprocesses: with JAX_ENABLE_X64=1 for the
# reference values, and with JAX_ENABLE_X64=0 for the calls that must raise
_X64_SCRIPT = """\
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax import enable_x64
except ImportError:  # older JAX releases
    from jax.experimental import enable_x64

from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
from s2fft.recursions import price_mcewen, turok_jax
from s2fft.sampling import s2_samples
from s2fft.transforms import otf_recursions, spherical, wigner

data = np.load(sys.argv[1])
L, N, spin = (int(data[key]) for key in ("L", "N", "spin"))
thetas = s2_samples.thetas(L, "gl")
x64 = jax.config.read("jax_enable_x64")


def cases(single):
    def get(key):
        value = data[key]
        if single:
            value = value.astype(np.complex64 if np.iscomplexobj(value) else np.float32)
        return value

    flm, f, flmn, fw, ftm, weights = map(
        get, ("flm", "f", "flmn", "fw", "ftm", "weights")
    )

    def inverse(flm):
        return spherical.inverse_jax(flm, L, spin=spin, sampling="mw")

    return {
        "inverse_jax": lambda: inverse(flm),
        "forward_jax": lambda: spherical.forward_jax(f, L, spin=spin, sampling="mw"),
        "auto_kernel": lambda: spin_spherical_kernel_jax(L, spin, recursion="auto"),
        "wigner_inverse_jax": lambda: wigner.inverse_jax(flmn, L, N),
        "wigner_forward_jax": lambda: wigner.forward_jax(fw, L, N),
        "grad_inverse_jax": lambda: jax.grad(
            lambda x: jnp.sum(weights * jnp.real(inverse(x)))
        )(flm),
        "vmap_inverse_jax": lambda: jax.vmap(inverse)(jnp.stack([flm, 2 * flm])),
        "user_jit_inverse_jax": lambda: jax.jit(inverse)(flm),
        "reality_forward_jax": lambda: spherical.forward_jax(
            np.real(f), L, spin=spin, sampling="mw", reality=True
        ),
        "generate_precomputes_jax": lambda: price_mcewen.generate_precomputes_jax(
            L, spin, "mw"
        ),
        "turok_jax_compute_slice": lambda: turok_jax.compute_slice(
            thetas[1], L - 1, L, -spin
        ),
        "inverse_latitudinal_step_jax": lambda: (
            otf_recursions.inverse_latitudinal_step_jax(
                flm, thetas, L, spin, None, "gl"
            )
        ),
        "forward_latitudinal_step_jax": lambda: (
            otf_recursions.forward_latitudinal_step_jax(
                ftm, thetas, L, spin, None, "gl"
            )
        ),
    }


def run(calls, suffix=""):
    for name, call in calls.items():
        try:
            leaves = jax.tree_util.tree_leaves(call())
            values[name + suffix] = np.concatenate([np.ravel(v) for v in leaves])
        except Exception as error:
            errors[name + suffix] = f"{type(error).__name__}: {error}"


values, errors = {}, {}
if x64:
    run(cases(single=False))
else:
    run(cases(single=True))
    # The same calls in double precision, with x64 enabled only inside a context, to
    # check that the calls above left no single precision state behind
    with enable_x64(True):
        run(cases(single=False), "_x64")
np.savez(sys.argv[2], **values)
print(json.dumps({"errors": errors, "x64": jax.config.read("jax_enable_x64")}))
"""


def test_otf_transforms_x64_disabled(tmp_path, rng: np.random.Generator):
    """
    Test that the JAX Wigner-d recursions refuse to run with 64-bit mode disabled.

    In single precision the recursions overflowed and returned mostly zeros, without an
    error. With x64 disabled, every JAX path through them tested here must raise a
    RuntimeError that names jax_enable_x64, and leave x64 disabled; the same calls
    inside jax.enable_x64 must then match their results with x64 enabled throughout,
    so that no single precision state is left behind. The calls run in subprocesses,
    so that disabling x64 cannot leak into other tests.
    """
    L, N, spin = 16, 2, 2
    flm = signal_generator.generate_flm(rng, L, spin=spin)
    flmn = signal_generator.generate_flmn(rng, L, N)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        L=L,
        N=N,
        spin=spin,
        flm=flm,
        f=np.asarray(spherical.inverse_jax(flm, L, spin=spin, sampling="mw")),
        flmn=flmn,
        fw=np.asarray(wigner.inverse_jax(flmn, L, N)),
        ftm=rng.standard_normal(samples.ftm_shape(L, "gl"))
        + 1j * rng.standard_normal(samples.ftm_shape(L, "gl")),
        weights=rng.standard_normal(samples.f_shape(L, "mw")),
    )

    def run(x64):
        outputs = tmp_path / f"outputs_{x64}.npz"
        result = subprocess.run(  # noqa: S603 (a fixed script)
            [sys.executable, "-c", _X64_SCRIPT, str(data), str(outputs)],
            env={**os.environ, "JAX_ENABLE_X64": "1" if x64 else "0"},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout.splitlines()[-1])
        assert report["x64"] is x64, "x64 was changed"
        return report["errors"], np.load(outputs)

    reference_errors, references = run(x64=True)
    assert not reference_errors, reference_errors
    errors, values = run(x64=False)

    def relative_error(value, reference):
        return np.max(np.abs(value - reference)) / np.max(np.abs(reference))

    problems = []
    for name in references.files:
        reference = references[name]
        message = errors.get(name)
        if message is None:
            value = values[name]
            problems.append(
                f"{name} returned values with x64 disabled (relative error "
                f"{relative_error(value, reference):.3g}, "
                f"{np.mean(value == 0):.0%} zeros)"
            )
        elif not re.match(r"RuntimeError: .*jax_enable_x64", message, re.DOTALL):
            problems.append(f"{name} raised with x64 disabled: {message}")
        if name + "_x64" in errors:
            message = errors[name + "_x64"]
            problems.append(f"{name} raised inside jax.enable_x64: {message}")
        else:
            error = relative_error(values[name + "_x64"], reference)
            if not error < 1e-12:
                problems.append(
                    f"{name} inside jax.enable_x64: relative error {error:.3g}"
                )
    assert len(references.files) == 13
    assert not problems, "; ".join(problems)
