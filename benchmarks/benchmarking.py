"""Helper functions and classes for benchmarking.

Functions to be benchmarked in a module should be decorated with `benchmark` which
takes one positional argument corresponding to a function to peform any necessary
set up for the benchmarked function (returning a dictionary, potentially empty with
any precomputed values to pass to benchmark function as keyword arguments) and zero
or more keyword arguments specifying parameter names and values lists to benchmark
over (the Cartesian product of all specified parameter values is used). The benchmark
function is passed the union of any precomputed values outputted by the setup function
and the parameters values as keyword arguments.

As a simple example, the following defines a benchmark for computing the mean of a list
of numbers.

```Python
import random
from benchmarking import benchmark

def setup_mean(n):
    return {"x": [random.random() for _ in range(n)]}, None

@benchmark(setup_computation, n=[1, 2, 3, 4])
def mean(x, n):
    return sum(x) / n
```

The `skip` function can be used to skip the benchmark for certain parameter values.
For example

```Python
import random
from benchmarking import benchmark, skip

def setup_mean(n):
    return {"x": [random.random() for _ in range(n)]}, None

@benchmark(setup_computation, n=[0, 1, 2, 3, 4])
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

import argparse
import datetime
import inspect
import json
import platform
import timeit
from ast import literal_eval
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class SkipBenchmarkException(Exception):
    """Exception to be raised to skip benchmark for some parameter set."""


def _get_version_or_none(package_name):
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


def _get_gpu_memory_mebibytes(device):
    """Try to get GPU memory available in mebibytes (MiB)."""
    memory_stats = device.memory_stats()
    if memory_stats is None:
        return None
    bytes_limit = memory_stats.get("bytes_limit")
    return bytes_limit // 2**20 if bytes_limit is not None else None


def _get_gpu_info():
    """Get details of GPU devices available from JAX or None if JAX not available."""
    try:
        import jax

        return [
            {
                "kind": d.device_kind,
                "memory_available / MiB": _get_gpu_memory_mebibytes(d),
            }
            for d in jax.devices()
            if d.platform == "gpu"
        ]
    except ImportError:
        return None


def _get_cuda_info():
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


def skip(message):
    """Skip benchmark for a particular parameter set with explanatory message.

    Args:
        message (str): Message explaining why benchmark parameter set was skipped.
    """
    raise SkipBenchmarkException(message)


def benchmark(setup_=None, **parameters):
    """Decorator for defining a function to be benchmark.

    Args:
        setup_: Function performing any necessary set up for benchmark, and the resource
            usage of which will not be tracked in benchmarking. The function should
            return a 2-tuple, with first entry a dictionary of values to pass to the
            benchmark as keyword arguments, corresponding to any precomputed values,
            and the second entry optionally a reference value specifying the expected
            'true' numerical output of the behchmarked function to allow computing
            numerical error, or `None` if there is no relevant reference value.

    Kwargs:
        Parameter names and associated lists of values over which to run benchmark.
        The benchmark is run for the Cartesian product of all parameter values.

    Returns:
        Decorator which marks function as benchmark and sets setup function and
        parameters attributes. To also record error of benchmarked function in terms of
        maximum absolute elementwise difference between output and reference value
        returned by `setup_` function, the decorated benchmark function should return
        the numerical value to measure the error for.
    """

    def decorator(function):
        function.is_benchmark = True
        function.setup = setup_ if setup_ is not None else lambda: {}
        function.parameters = parameters
        return function

    return decorator


def _parameters_string(parameters):
    """Format parameter values as string for printing benchmark results."""
    return "(" + ", ".join(f"{name}: {val}" for name, val in parameters.items()) + ")"


def _dict_product(dicts):
    """Generator corresponding to Cartesian product of dictionaries."""
    return (dict(zip(dicts.keys(), values)) for values in product(*dicts.values()))


def _parse_value(value):
    """Parse a value passed at command line as a Python literal or string as fallback"""
    try:
        return literal_eval(value)
    except ValueError:
        return str(value)


def _parse_parameter_overrides(parameter_overrides):
    """Parse any parameter override values passed as command line arguments"""
    return (
        {
            parameter: [_parse_value(v) for v in values]
            for parameter, *values in parameter_overrides
        }
        if parameter_overrides is not None
        else {}
    )


def _parse_cli_arguments(description):
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


def _is_benchmark(object):
    """Predicate for testing whether an object is a benchmark function or not."""
    return (
        inspect.isfunction(object)
        and hasattr(object, "is_benchmark")
        and object.is_benchmark
    )


def collect_benchmarks(module, benchmark_names):
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


def run_benchmarks(
    benchmarks,
    number_runs,
    number_repeats,
    print_results=True,
    parameter_overrides=None,
):
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
                precomputes, reference_output = benchmark.setup(**parameter_set)
                benchmark_function = partial(benchmark, **precomputes, **parameter_set)
                # Run benchmark once without timing to record output for potentially
                # computing numerical error and to remove effect of any one-off costs
                # such as just-in-time compilation when timing
                output = benchmark_function()
                run_times = [
                    time / number_runs
                    for time in timeit.repeat(
                        benchmark_function, number=number_runs, repeat=number_repeats
                    )
                ]
                results_entry = {**parameter_set, "times / s": run_times}
                if reference_output is not None and output is not None:
                    results_entry["error"] = abs(reference_output - output).max()
                if MEMORY_PROFILER_AVAILABLE:
                    baseline_memory = memory_profiler.memory_usage(max_usage=True)
                    peak_memory = (
                        memory_profiler.memory_usage(
                            benchmark_function,
                            interval=max(run_times) * number_repeats,
                            max_usage=True,
                            max_iterations=number_repeats,
                            include_children=True,
                        )
                        - baseline_memory
                    )
                    results_entry["peak_memory / MiB"] = peak_memory
                results[benchmark.__name__].append(results_entry)
                if print_results:
                    print(
                        (
                            f"{_parameters_string(parameter_set):>40}: \n    "
                            if len(parameter_set) != 0
                            else "    "
                        )
                        + f"min(time): {min(run_times):>#7.2g}s, "
                        + f"max(time): {max(run_times):>#7.2g}s"
                        + (
                            f", peak mem.: {peak_memory:>#7.2g}MiB"
                            if MEMORY_PROFILER_AVAILABLE
                            else ""
                        )
                        + (
                            f", max(abs(error)): {results_entry['error']:#7.2g}"
                            if "error" in results_entry
                            else ""
                        )
                    )
            except SkipBenchmarkException as e:
                if print_results:
                    print(
                        f"{_parameters_string(parameter_set):>40}: skipped - {str(e)}"
                    )
    return results


def parse_args_collect_and_run_benchmarks(module=None):
    """Collect and run all benchmarks in a module and parse command line arguments.

    Args:
        module: Module containing benchmarks to run. Defaults to module from which
            this function was called if not specified (set to `None`).

    Returns:
        Dictionary containing timing (and potentially memory usage) results for each
        parameters set of each benchmark function.
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
        parameter_overrides=parameter_overrides,
    )
    if args.output_file is not None:
        package_versions = {
            f"{package}_version": _get_version_or_none(package)
            for package in ("s2fft", "jax", "numpy")
        }
        system_info = {
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
        with open(args.output_file, "w") as f:
            output = {
                "date_time": datetime.datetime.now().isoformat(),
                "benchmark_module": module.__name__,
                "system_info": system_info,
                "results": results,
            }
            json.dump(output, f, indent=True)
    return results
