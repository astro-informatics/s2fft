"""Helper functions and classes for benchmarking.

Functions to be benchmarked in a module should be decorated with `benchmark` which
takes one positional argument corresponding to a function to peform any necessary
set up for the benchmarked function (returning a dictionary, potentially empty with
any precomputed values to pass to benchmark function as keyword arguments) and zero
or more keyword arguments specifying parameter names and values lists to benchmark
over (the Cartesian product of all specified parameter values is used). The benchmark
function is passed the union of any precomputed values outputted by the setup function
and the parameters values as keyword arguments.

As a simple example, the following defines a benchmarkfor computing the mean of a list
of numbers.

```Python
import random
from benchmarking import benchmark

def setup_mean(n):
    return {"x": [random.random() for _ in range(n)]}

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
    return {"x": [random.random() for _ in range(n)]}

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
from ast import literal_eval
from functools import partial
from itertools import product
from pathlib import Path
import json
import timeit
import inspect

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class SkipBenchmarkException(Exception):
    """Exception to be raised to skip benchmark for some parameter set."""


def skip(message):
    """Skip benchmark for a particular parameter set with explanatory message.

    Args:
        message (str): Message explaining why benchmark parameter set was skipped.
    """
    raise SkipBenchmarkException(message)


def benchmark(setup_=None, **parameters):
    """Decorator for defining a function to be benchmarker

    Args:
        setup_: Function performing any necessary set up for benchmark, and the resource
            usage of which will not be tracked in benchmarking. The function should
            return a dictionary of values to pass to the benchmark as keyword arguments.

    Kwargs:
        Parameter names and associated lists of values over which to run benchmark.
        The benchmark is run for the Cartesian product of all parameter values.

    Returns:
        Decorator which marks function as benchmark and sets setup function and
        parameters attributes.
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


def _parse_cli_arguments():
    """Parse command line arguments passed for controlling benchmark runs"""
    parser = argparse.ArgumentParser("Run benchmarks")
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
    return parser.parse_args()


def _is_benchmark(object):
    """Predicate for testing whether an object is a benchmark function or not."""
    return (
        inspect.isfunction(object)
        and hasattr(object, "is_benchmark")
        and object.is_benchmark
    )


def collect_benchmarks(module):
    """Collect all benchmark functions from a module.

    Args:
        module: Python module containing benchmark functions.

    Returns:
        List of functions in module with `is_benchmark` attribute set to `True`.
    """
    return [function for name, function in inspect.getmembers(module, _is_benchmark)]


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
        results[benchmark.__name__] = {}
        if print_results:
            print(benchmark.__name__)
        parameters = benchmark.parameters.copy()
        if parameter_overrides is not None:
            parameters.update(parameter_overrides)
        for parameter_set in _dict_product(parameters):
            try:
                precomputes = benchmark.setup(**parameter_set)
                benchmark_function = partial(benchmark, **precomputes, **parameter_set)
                run_times = [
                    time / number_runs
                    for time in timeit.repeat(
                        benchmark_function, number=number_runs, repeat=number_repeats
                    )
                ]
                results[benchmark.__name__] = {**parameter_set, "times / s": run_times}
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
                    results[benchmark.__name__]["peak_memory / MiB"] = peak_memory
                if print_results:
                    print(
                        (
                            f"{_parameters_string(parameter_set):>40}: \n    "
                            if len(parameter_set) != 0
                            else "    "
                        )
                        + f"min(time): {min(run_times):>#7.2g}s, "
                        + f"max(time): {max(run_times):>#7.2g}s, "
                        + (
                            f"peak mem.: {peak_memory:>#7.2g}MiB"
                            if MEMORY_PROFILER_AVAILABLE
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
    args = _parse_cli_arguments()
    parameter_overrides = _parse_parameter_overrides(args.parameter_overrides)
    if module is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
    results = run_benchmarks(
        benchmarks=collect_benchmarks(module),
        number_runs=args.number_runs,
        number_repeats=args.repeats,
        parameter_overrides=parameter_overrides,
    )
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            json.dump(results, f)
    return results
