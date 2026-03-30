import jax
import numpy as np
import pytest
import torch

import s2fft
from s2fft.precompute_transforms.spherical import forward, inverse
from s2fft.utils._dtype_association import compatible_complex_dtype

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
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
