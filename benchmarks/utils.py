"""
Common utility functions
"""

from functools import partial
from itertools import product
import timeit


def parametrize(parameter_dict):
    """
    Takes a function and adds a dictionary's keys and values as attributes to it
    """

    def decorator(function):
        function.param_names = list(parameter_dict.keys())
        function.params = list(parameter_dict.values())
        return function

    return decorator


def parameters_string(parameters, names):
    return (
        "(" + ", ".join(f"{name}: {val}" for name, val in zip(names, parameters)) + ")"
    )


def run_benchmarks(benchmarks, number_runs, number_repeats, print_results=True):
    results = {}
    for benchmark in benchmarks:
        results[benchmark.__name__] = {}
        if print_results:
            print(benchmark.__name__)
        for parameters in product(*benchmark.params):
            parameters_key = tuple(zip(benchmark.param_names, parameters))
            results[benchmark.__name__][parameters_key] = {}
            benchmark_function = partial(benchmark, *parameters)
            run_times = [
                time / number_runs
                for time in timeit.repeat(
                    benchmark_function, number=number_runs, repeat=number_repeats
                )
            ]
            results[benchmark.__name__][parameters_key]["time"] = run_times
            if print_results:
                print(
                    f"{parameters_string(parameters, benchmark.param_names):>40}: "
                    f"min(time): {min(run_times):>#7.2g}s, "
                    f"max(time): {max(run_times):>#7.2g}s"
                )
    return results
