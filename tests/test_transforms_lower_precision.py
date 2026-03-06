import jax
import numpy as np
import pytest
import torch

from s2fft import inverse as precise_inverse
from s2fft.precompute_transforms.spherical import _kernel_functions, forward, inverse
from s2fft.utils._dtype_association import compatible_cmplx_dtype

jax.config.update("jax_enable_x64", True)


# TODO: recycled from test_spherical_transforms_precompute - refactor into a fixture perhaps?
def get_flm_and_kernel(
    flm_generator,
    L,
    spin,
    sampling,
    reality,
    method,
    recursion,
    forward,
    nside=None,
):
    flm = flm_generator(L=L, spin=spin, reality=reality)
    kfunc = _kernel_functions[method]
    kernel = kfunc(L, spin, reality, sampling, nside, forward, recursion=recursion)
    return flm, kernel


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
@pytest.mark.parametrize("reality", [True, False])
@pytest.mark.parametrize("method", ["jax", "torch"])
def test_forward_lower_precision(
    flm_generator,
    sampling: str,
    method: str,
    reality: bool,
    L: int = 64,
    spin: int = 0,
    recursion: str = "auto",
):
    """
    Verify that flm coefficients inherit the dtype of the input arrays.

    Test is run as a matrix across:
    - sampling (to ensure there are no code-paths that are still forcing array creation with a fixed dtype)
    - reality (effectively handles the two cases for real signals and complex signals)
    - FIXME method (ensure that dtype behaviour occurs for all of the numpy / jax / torch paths)
    """
    nside = L // 2 if sampling == "healpix" else None

    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=True,
        nside=nside,
    )
    f = precise_inverse(
        flm,
        L=L,
        spin=spin,
        nside=nside,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    flm_recovered_long = forward(
        f,
        L=L,
        spin=spin,
        nside=nside,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
    )

    short_dtype = "float32" if reality else "complex64"
    casting_method = "astype"
    if method == "torch":
        short_dtype = getattr(torch, short_dtype)
        casting_method = "to"

    f_lower_precision = getattr(f, casting_method)(short_dtype)
    kernel_lower_precision = getattr(kernel, casting_method)(short_dtype)
    expected_short_flm_type = compatible_cmplx_dtype(f_lower_precision)

    flm_recovered_short = forward(
        f_lower_precision,
        L=L,
        spin=spin,
        nside=nside,
        kernel=kernel_lower_precision,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    flm_recovered_short_dtype = flm_recovered_short.dtype
    if method == "torch":
        flm = torch.Tensor(flm)
        flm_recovered_short_dtype = str(flm_recovered_short_dtype)

    round_trip_error_long_dtype = abs(flm - flm_recovered_long).max()
    long_dtype_error_oom = np.round(np.log10(round_trip_error_long_dtype))

    round_trip_error_short_dtype = abs(flm - flm_recovered_short).max()
    short_dtype_error_oom = np.round(np.log10(round_trip_error_short_dtype))

    # Confirm that the output inherits the lower precision dtype
    assert flm_recovered_short_dtype == expected_short_flm_type
    # mw and mwss currently fails for numpy runs due to this:
    # FIXME https://github.com/numpy/numpy/issues/17801!
    # Fixed in numpy 2.0.0 but we seem to be pinned to a numpy v1.XX

    # Naive expectations for the error. 1/2 precision ~= 1/2 the error OOMagnitude.
    # Allow a -/+1 margin for near-misses during rounding and taking log.
    assert (
        long_dtype_error_oom <= 2 * short_dtype_error_oom
        or long_dtype_error_oom == pytest.approx(2 * short_dtype_error_oom, abs=1)
    )


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
@pytest.mark.parametrize("reality", [True, False])
@pytest.mark.parametrize("method", ["jax"])
def test_inverse_lower_precision(
    flm_generator,
    sampling: str,
    method: str,
    reality: bool,
    L: int = 64,
    spin: int = 0,
    recursion: str = "auto",
) -> None:
    """
    Verify that signal arrays inherit the dtype of the input harmonic coefficient arrays.

    Test is run as a matrix across:
    - sampling (to ensure there are no code-paths that are still forcing array creation with a fixed dtype)
    - reality (effectively handles the two cases for real signals and complex signals). Note that for the inverse transform, the output is returned as a floatXX array if reality is set to True, however this conversion doesn't
    actually occur until after all computations have been conducted.
    - FIXME method (ensure that dtype behaviour occurs for all of the numpy / jax / torch paths)
    """
    nside = L // 2 if sampling == "healpix" else None

    flm, kernel = get_flm_and_kernel(
        flm_generator,
        L,
        spin,
        sampling,
        reality,
        method,
        recursion,
        forward=False,
        nside=nside,
    )
    trusted_signal = precise_inverse(
        flm,
        L=L,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
    )

    f_recovered_long = inverse(
        flm,
        L=L,
        kernel=kernel,
        sampling=sampling,
        reality=reality,
        method=method,
        nside=nside,
    )

    short_dtype = "complex64"
    expected_short_f_type = "float32" if reality else "complex64"
    casting_method = "astype"
    if method == "torch":
        short_dtype = getattr(torch, short_dtype)
        casting_method = "to"
        expected_short_f_type = getattr(torch, expected_short_f_type)

    flm_lower_precision = getattr(flm, casting_method)(short_dtype)
    kernel_lower_precision = getattr(kernel, casting_method)(short_dtype)

    f_recovered_short = inverse(
        flm_lower_precision,
        L=L,
        spin=spin,
        nside=nside,
        kernel=kernel_lower_precision,
        sampling=sampling,
        reality=reality,
        method=method,
    )
    f_recovered_short_dtype = f_recovered_short.dtype
    if method == "torch":
        f_recovered_short_dtype = str(f_recovered_short_dtype)

    error_long_dtype = abs(trusted_signal - f_recovered_long).max()
    long_dtype_error_oom = np.round(np.log10(error_long_dtype))

    round_trip_error_short_dtype = abs(trusted_signal - f_recovered_short).max()
    short_dtype_error_oom = np.round(np.log10(round_trip_error_short_dtype))

    # Confirm that the output inherits the lower precision dtype
    assert f_recovered_short_dtype == expected_short_f_type
    # mw and mwss currently fails for numpy runs due to this:
    # FIXME https://github.com/numpy/numpy/issues/17801!
    # Fixed in numpy 2.0.0 but we seem to be pinned to a numpy v1.XX

    # Naive expectations for the error. 1/2 precision ~= 1/2 the error OOMagnitude.
    # Allow a -/+1 margin for near-misses during rounding and taking log.
    assert (
        long_dtype_error_oom <= 2 * short_dtype_error_oom
        or long_dtype_error_oom == pytest.approx(2 * short_dtype_error_oom, abs=1)
    )
