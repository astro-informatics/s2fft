import json
import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import s2fft
from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
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


# Run by test_otf_transforms_x64_disabled in a subprocess with JAX_ENABLE_X64=0
_X64_DISABLED_SCRIPT = """\
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

assert not jax.config.read("jax_enable_x64")

from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
from s2fft.transforms import spherical, wigner

inputs = np.load(sys.argv[1])
L, N, spin = (int(inputs[key]) for key in ("L", "N", "spin"))
flm = inputs["flm"].astype(np.complex64)
f = inputs["f"].astype(np.complex64)
flmn = inputs["flmn"].astype(np.complex64)
weights = inputs["weights"].astype(np.float32)


def inverse(flm):
    return spherical.inverse_jax(flm, L, spin=spin, sampling="mw")


cases = {
    "inverse_jax": lambda: inverse(flm),
    "forward_jax": lambda: spherical.forward_jax(f, L, spin=spin, sampling="mw"),
    "auto_kernel": lambda: spin_spherical_kernel_jax(L, spin, recursion="auto"),
    "wigner_inverse_jax": lambda: wigner.inverse_jax(flmn, L, N),
    "grad_inverse_jax": lambda: jax.grad(
        lambda x: jnp.sum(weights * jnp.real(inverse(x)))
    )(flm),
    "vmap_inverse_jax": lambda: jax.vmap(inverse)(jnp.stack([flm, 2 * flm])),
}
values, errors = {}, {}
for name, case in cases.items():
    try:
        values[name] = np.asarray(case())
    except Exception as error:
        errors[name] = f"{type(error).__name__}: {error}"
# The same calls in double precision, with x64 enabled only inside a context, to check
# that the calls above left no single precision state behind
flm, f, flmn, weights = (inputs[key] for key in ("flm", "f", "flmn", "weights"))
# jax.enable_x64 is jax.experimental.enable_x64 in older JAX versions
enable_x64 = getattr(jax, "enable_x64", None)
if enable_x64 is None:
    from jax.experimental import enable_x64
with enable_x64(True):
    for name, case in cases.items():
        try:
            values[name + "_x64"] = np.asarray(case())
        except Exception as error:
            errors[name + "_x64"] = f"{type(error).__name__}: {error}"
np.savez(sys.argv[2], **values)
print(json.dumps({"errors": errors, "x64": jax.config.read("jax_enable_x64")}))
"""


def test_otf_transforms_x64_disabled(tmp_path, rng: np.random.Generator):
    """
    Test that the on-the-fly transforms refuse to run with JAX's 64-bit mode disabled.

    In single precision the Wigner-d recursions overflowed and returned mostly zeros,
    without an error. With x64 disabled, each call must raise a RuntimeError that names
    jax_enable_x64, and leave x64 disabled. The same calls inside jax.enable_x64 must
    then match their double precision results, so that no single precision state is
    left behind. The calls run in a subprocess, so that disabling x64 cannot leak into
    other tests.
    """
    L, N, spin = 16, 2, 2
    flm = signal_generator.generate_flm(rng, L, spin=spin)
    flmn = signal_generator.generate_flmn(rng, L, N)
    weights = rng.standard_normal(samples.f_shape(L, "mw"))

    def inverse(flm):
        return spherical.inverse_jax(flm, L, spin=spin, sampling="mw")

    f = np.asarray(inverse(flm))
    references = {
        "inverse_jax": f,
        "forward_jax": spherical.forward_jax(f, L, spin=spin, sampling="mw"),
        "auto_kernel": spin_spherical_kernel_jax(L, spin, recursion="auto"),
        "wigner_inverse_jax": wigner.inverse_jax(flmn, L, N),
        "grad_inverse_jax": jax.grad(lambda x: jnp.sum(weights * jnp.real(inverse(x))))(
            flm
        ),
        "vmap_inverse_jax": jax.vmap(inverse)(jnp.stack([flm, 2 * flm])),
    }
    inputs, outputs = tmp_path / "inputs.npz", tmp_path / "outputs.npz"
    np.savez(inputs, L=L, N=N, spin=spin, flm=flm, f=f, flmn=flmn, weights=weights)
    result = subprocess.run(  # noqa: S603 (a fixed script)
        [sys.executable, "-c", _X64_DISABLED_SCRIPT, str(inputs), str(outputs)],
        env={**os.environ, "JAX_ENABLE_X64": "0"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["x64"] is False, "x64 was left enabled"
    values = np.load(outputs)

    def relative_error(value, reference):
        return np.max(np.abs(value - reference)) / np.max(np.abs(reference))

    problems = []
    for name, reference in references.items():
        message = report["errors"].get(name)
        if message is None:
            value = values[name]
            problems.append(
                f"{name} returned values with x64 disabled (relative error "
                f"{relative_error(value, reference):.3g}, "
                f"{np.mean(value == 0):.0%} zeros)"
            )
        elif not re.match(r"RuntimeError: .*jax_enable_x64", message, re.DOTALL):
            problems.append(f"{name} raised with x64 disabled: {message}")
        if name + "_x64" in report["errors"]:
            message = report["errors"][name + "_x64"]
            problems.append(f"{name} raised inside jax.enable_x64: {message}")
        else:
            error = relative_error(values[name + "_x64"], reference)
            if not error < 1e-12:
                problems.append(
                    f"{name} inside jax.enable_x64: relative error {error:.3g}"
                )
    assert not problems, "; ".join(problems)
