"""Helper functions and classes for benchmarking.

Functions to be benchmarked in a module should be decorated with `benchmark` which takes
one positional argument corresponding to a function to perform any necessary set up for
the benchmarked function and zero or more keyword arguments specifying parameter names
and values lists to benchmark over (the Cartesian product of all specified parameter
values is used).

The benchmark setup function should return an instance of the named tuple type
`BenchmarkSetup` which consists of a required dictionary entry, potentially empty with
any precomputed values to pass to benchmark function as keyword arguments, and two
further optional entries: the first a reference value for the output of the benchmarked
function to use to compute numerical error if applicable, defaulting to `None`
indicating no applicable reference value; and the second a flag indicating whether to
just-in-time compile the benchmark function using JAX's `jit` transform, defaulting to
`False`.

The benchmark function is passed the union of any precomputed values returned by the
setup function and the parameters values as keyword arguments. If a reference output
value is set by the setup function the benchmark function should output the value to
compare to this reference value by computing the maximum absolute elementwise
difference. If the function is to be just-in-time compiled using JAX the value returned
by the benchmark function should be a JAX array on which the `block_until_ready` method
may be called to ensure the function only exits once the relevant computation has
completed (necessary due to JAX's asynchronous dispatch computation model).

As a simple example, the following defines a benchmark for computing the mean of a list
of numbers.

```Python
import random
from benchmarking import BenchmarkSetup, benchmark

def setup_mean(n):
    return BenchmarkSetup({"x": [random.random() for _ in range(n)]})

@benchmark(setup_mean, n=[1, 2, 3, 4])
def mean(x, n):
    return sum(x) / n
```

The `skip` function can be used to skip the benchmark for certain parameter values.
For example

```Python
import random
from benchmarking import BenchmarkSetup, benchmark, skip

def setup_mean(n):
    return BenchmarkSetup({"x": [random.random() for _ in range(n)]})

@benchmark(setup_mean, n=[0, 1, 2, 3, 4])
def mean(x, n):
    if n == 0:
        skip("number of items must be positive")
    return sum(x) / n
```

This makes most sense when excluding certain combinations of parameters.

The `parse_args_collect_and_run_benchmarks` function should be called within a
`if __name__ == '__main__'` block at the end of the module defining the benchmarks
to allow it to be executed as a script for runnning the benchmarks:

```Python
from benchmarking import benchmark, parse_args_collect_and_run_benchmarks

...

if __name__ == "__main__":
    parse_args_collect_and_run_benchmarks()
```
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import inspect
import json
import platform
import timeit
import tracemalloc
from ast import literal_eval
from collections.abc import Callable, Iterable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import jax
import numpy as np


class SkipBenchmarkException(Exception):
    """Exception to be raised to skip benchmark for some parameter set."""


def _get_version_or_none(package_name: str) -> str | None:
    """Get installed version of package or `None` if package not found."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _get_cpu_info():
    """Get details of CPU from cpuinfo if available or None if not."""
    try:
        import cpuinfo

        return cpuinfo.get_cpu_info()
    except ImportError:
        return None


def _get_gpu_memory_in_bytes(device: jax.Device) -> int | None:
    """Try to get GPU memory available in bytes."""
    memory_stats = device.memory_stats()
    if memory_stats is None:
        return None
    return memory_stats.get("bytes_limit")


def _get_gpu_info() -> dict[str, str | int]:
    """Get details of GPU devices available from JAX or None if JAX not available."""
    try:
        import jax

        return [
            {
                "kind": d.device_kind,
                "memory_available_in_bytes": _get_gpu_memory_in_bytes(d),
            }
            for d in jax.devices()
            if d.platform == "gpu"
        ]
    except ImportError:
        return None


def _get_cuda_info() -> dict[str, str]:
    """Try to get information on versions of CUDA libraries."""
    try:
        from jax._src.lib import cuda_versions

        if cuda_versions is None:
            return None
        return {
            "cuda_runtime_version": cuda_versions.cuda_runtime_get_version(),
            "cuda_runtime_build_version": cuda_versions.cuda_runtime_build_version(),
            "cudnn_version": cuda_versions.cudnn_get_version(),
            "cudnn_build_version": cuda_versions.cudnn_build_version(),
            "cufft_version": cuda_versions.cufft_get_version(),
            "cufft_build_version": cuda_versions.cufft_build_version(),
        }
    except ImportError:
        return None


def skip(message: str) -> None:
    """Skip benchmark for a particular parameter set with explanatory message.

    Args:
        message (str): Message explaining why benchmark parameter set was skipped.
    """
    raise SkipBenchmarkException(message)


class BenchmarkSetup(NamedTuple):
    """Structure containing data for setting up a benchmark function."""

    arguments: dict[str, Any]
    reference_output: None | jax.Array | np.ndarray = None
    jit_benchmark: bool = False


def benchmark(
    setup: Callable[..., BenchmarkSetup] | None = None, **parameters
) -> Callable:
    """Decorator for defining a function to be benchmark.

    Args:
        setup: Function performing any necessary set up for benchmark, and the resource
            usage of which will not be tracked in benchmarking. The function should
            return an instance of `BenchmarkSetup` named tuple, with first entry a
            dictionary of values to pass to the benchmark as keyword arguments,
            corresponding to any precomputed values, the second entry optionally a
            reference value specifying the expected 'true' numerical output of the
            benchmarked function to allow computing numerical error, or `None` if there
            is no relevant reference value and third entry a boolean flag indicating
            whether to use JAX's just-in-time compilation transform to benchmark
            function.

    Kwargs:
        Parameter names and associated lists of values over which to run benchmark.
        The benchmark is run for the Cartesian product of all parameter values.

    Returns:
        Decorator which marks function as benchmark and sets setup function and
        parameters attributes. To also record error of benchmarked function in terms of
        maximum absolute elementwise difference between output and reference value
        returned by `setup` function, the decorated benchmark function should return
        the numerical value to measure the error for.
    """

    def decorator(function):
        function.is_benchmark = True
        function.setup = setup if setup is not None else lambda: {}
        function.parameters = parameters
        return function

    return decorator


def _parameters_string(parameters: dict) -> str:
    """Format parameter values as string for printing benchmark results."""
    return "(" + ", ".join(f"{name}: {val}" for name, val in parameters.items()) + ")"


def _format_results_entry(results_entry: dict) -> str:
    """Format benchmark results entry as a string for printing."""
    return (
        (
            f"{_parameters_string(results_entry['parameters']):>40}: \n    "
            if len(results_entry["parameters"]) != 0
            else "    "
        )
        + f"min(run times): {min(results_entry['run_times_in_seconds']):>#7.2g}s, "
        + f"max(run times): {max(results_entry['run_times_in_seconds']):>#7.2g}s"
        + (
            f", peak memory: {results_entry['traced_memory_peak_in_bytes']:>#7.2g}B"
            if "traced_memory_peak_in_bytes" in results_entry
            else ""
        )
        + (
            f", max(abs(error)): {results_entry['max_abs_error']:>#7.2g}"
            if "max_abs_error" in results_entry
            else ""
        )
        + (
            f", floating point ops: {results_entry['cost_analysis']['flops']:>#7.2g}"
            f", mem access: {results_entry['cost_analysis']['bytes_accessed']:>#7.2g}B"
            if "cost_analysis" in results_entry
            else ""
        )
    )


def _dict_product(dicts: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    """Generator corresponding to Cartesian product of dictionaries."""
    return (dict(zip(dicts.keys(), values)) for values in product(*dicts.values()))


def _parse_value(value: str) -> Any:
    """Parse a value passed at command line as a Python literal or string as fallback"""
    try:
        return literal_eval(value)
    except ValueError:
        return str(value)


def _parse_parameter_overrides(parameter_overrides: list[str]) -> dict[str, Any]:
    """Parse any parameter override values passed as command line arguments"""
    return (
        {
            parameter: [_parse_value(v) for v in values]
            for parameter, *values in parameter_overrides
        }
        if parameter_overrides is not None
        else {}
    )


def _parse_cli_arguments(description: str) -> argparse.Namespace:
    """Parse command line arguments passed for controlling benchmark runs"""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-number-runs",
        type=int,
        default=10,
        help="Number of times to run the benchmark in succession in each timing run.",
    )
    parser.add_argument(
        "-repeats",
        type=int,
        default=3,
        help="Number of times to repeat the benchmark runs.",
    )
    parser.add_argument(
        "-parameter-overrides",
        type=str,
        action="append",
        nargs="*",
        help=(
            "Override for values to use for benchmark parameter. A parameter name "
            "followed by space separated list of values to use. May be specified "
            "multiple times to override multiple parameters. "
        ),
    )
    parser.add_argument(
        "-output-file", type=Path, help="File path to write JSON formatted results to."
    )
    parser.add_argument(
        "-benchmarks",
        nargs="+",
        help="Names of benchmark functions to run. All benchmarks are run if omitted.",
    )
    return parser.parse_args()


def _is_benchmark(object: Any) -> bool:
    """Predicate for testing whether an object is a benchmark function or not."""
    return (
        inspect.isfunction(object)
        and hasattr(object, "is_benchmark")
        and object.is_benchmark
    )


def collect_benchmarks(
    module: ModuleType, benchmark_names: list[str]
) -> list[Callable]:
    """Collect all benchmark functions from a module.

    Args:
        module: Python module containing benchmark functions.
        benchmark_names: List of benchmark names to collect or `None` if all benchmarks
            in module to be collected.

    Returns:
        List of functions in module with `is_benchmark` attribute set to `True`.
    """
    return [
        function
        for name, function in inspect.getmembers(module, _is_benchmark)
        if benchmark_names is None or name in benchmark_names
    ]


@contextlib.contextmanager
def trace_memory_allocations(n_frames: int = 1) -> Callable[[], tuple[int, int]]:
    """Context manager for tracing memory allocations in managed with block.

    Returns a thunk (zero-argument function) which can be called on exit from with block
    to get tuple of current size and peak size of memory blocks traced in bytes.

    Args:
        n_frames: Limit on depth of frames to trace memory allocations in.

    Returns:
        A thunk (zero-argument function) which can be called on exit from with block to
        get tuple of current size and peak size of memory blocks traced in bytes.
    """
    tracemalloc.start(n_frames)
    current_size, peak_size = None, None
    try:
        yield lambda: (current_size, peak_size)
        current_size, peak_size = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()


def _compile_jax_benchmark_and_analyse(
    benchmark_function: Callable, results_entry: dict
) -> Callable:
    """Compile a JAX benchmark function and extract cost estimates if available."""
    compiled_benchmark_function = jax.jit(benchmark_function).lower().compile()
    cost_analysis = compiled_benchmark_function.cost_analysis()
    if cost_analysis is not None:
        if isinstance(cost_analysis, list):
            cost_analysis = cost_analysis[0]
        results_entry["cost_analysis"] = {
            "flops": cost_analysis.get("flops"),
            "bytes_accessed": cost_analysis.get("bytes accessed"),
        }
    memory_analysis = compiled_benchmark_function.memory_analysis()
    if memory_analysis is not None:
        results_entry["memory_analysis"] = {
            prefix + base_key: getattr(memory_analysis, prefix + base_key, None)
            for prefix in ("", "host_")
            for base_key in (
                "alias_size_in_bytes",
                "argument_size_in_bytes",
                "generated_code_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
            )
        }
    # Ensure block_until_ready called on benchmark output due to JAX's asynchronous
    # dispatch model: https://jax.readthedocs.io/en/latest/async_dispatch.html
    return lambda: compiled_benchmark_function().block_until_ready()


def run_benchmarks(
    benchmarks: list[Callable],
    number_runs: int,
    number_repeats: int,
    print_results: bool = True,
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a set of benchmarks.

    Args:
        benchmarks: Benchmark functions to run with `setup` and `parameter` attributes
            specifying setup function and parameters set.
        number_of_runs: Number of times to run the benchmark in succession in each
            timing run. Larger values will reduce noise but be slower to run.
        number_repeats: Number of repeats of timing runs of benchmark. Larger values
            will give more recorded values to characterise spread but be slower to run.
        print_results: Whether to print benchmark results to stdout.
        parameter_overrides: Dictionary specifying any overrides for parameter values
            set in `benchmark` decorator.

    Returns:
        Dictionary containing timing (and potentially memory usage) results for each
        parameters set of each benchmark function.
    """
    results = {}
    for benchmark in benchmarks:
        results[benchmark.__name__] = []
        if print_results:
            print(benchmark.__name__)
        parameters = benchmark.parameters.copy()
        if parameter_overrides is not None:
            for parameter_name, parameter_values in parameter_overrides.items():
                if parameter_name in parameters:
                    parameters[parameter_name] = parameter_values
        for parameter_set in _dict_product(parameters):
            try:
                args, reference_output, jit_benchmark = benchmark.setup(**parameter_set)
                benchmark_function = partial(benchmark, **args, **parameter_set)
                results_entry = {"parameters": parameter_set}
                if jit_benchmark:
                    benchmark_function = _compile_jax_benchmark_and_analyse(
                        benchmark_function, results_entry
                    )
                # Run benchmark once without timing to record output for potentially
                # computing numerical error and trace memory usage
                with trace_memory_allocations() as traced_memory:
                    output = benchmark_function()
                current_size, peak_size = traced_memory()
                results_entry["traced_memory_final_in_bytes"] = current_size
                results_entry["traced_memory_peak_in_bytes"] = peak_size
                if reference_output is not None and output is not None:
                    results_entry["max_abs_error"] = float(
                        abs(reference_output - output).max()
                    )
                    results_entry["mean_abs_error"] = float(
                        abs(reference_output - output).mean()
                    )
                run_times = [
                    time / number_runs
                    for time in timeit.repeat(
                        benchmark_function, number=number_runs, repeat=number_repeats
                    )
                ]
                results_entry["run_times_in_seconds"] = run_times
                results[benchmark.__name__].append(results_entry)
                if print_results:
                    print(_format_results_entry(results_entry))
            except SkipBenchmarkException as e:
                if print_results:
                    print(
                        f"{_parameters_string(parameter_set):>40}: skipped - {str(e)}"
                    )
    return results


def get_system_info() -> dict[str, Any]:
    """Get dictionary of metadata about system.

    Returns:
        Dictionary with information about system, CPU and GPU devices (if present) and
        Python environment and package versions.
    """
    package_versions = {
        f"{package}_version": _get_version_or_none(package)
        for package in ("s2fft", "jax", "numpy")
    }
    return {
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "node": platform.node(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "release": platform.release(),
        "system": platform.system(),
        "cpu_info": _get_cpu_info(),
        "gpu_info": _get_gpu_info(),
        "cuda_info": _get_cuda_info(),
        **package_versions,
    }


def write_json_results_file(
    output_file: Path, results: dict[str, Any], benchmark_module: str
) -> None:
    """Write benchmark results and system information to a file in JSON format.

    Args:
        output_file: Path to file to write results to.
        results: Dictionary of benchmark results from `run_benchmarks`.
        benchmarks_module: Name of module containing benchmarks.
    """
    with open(output_file, "w") as f:
        output = {
            "date_time": datetime.datetime.now().isoformat(),
            "benchmark_module": benchmark_module,
            "system_info": get_system_info(),
            "results": results,
        }
        json.dump(output, f, indent=True)


def parse_args_collect_and_run_benchmarks(module: ModuleType | None = None) -> None:
    """Collect and run all benchmarks in a module and parse command line arguments.

    Args:
        module: Module containing benchmarks to run. Defaults to module from which
            this function was called if not specified (set to `None`).
    """
    if module is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
    args = _parse_cli_arguments(module.__doc__)
    parameter_overrides = _parse_parameter_overrides(args.parameter_overrides)
    results = run_benchmarks(
        benchmarks=collect_benchmarks(module, args.benchmarks),
        number_runs=args.number_runs,
        number_repeats=args.repeats,
        print_results=True,
        parameter_overrides=parameter_overrides,
    )
    if args.output_file is not None:
        write_json_results_file(args.output_file, results, module.__name__)
