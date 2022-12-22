"""
Common utility functions
"""

from functools import partial
from itertools import product
import timeit
from collections import defaultdict


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


def print_summary(results):

    print(f"\n--- Summary of benchmarks")
    number_of_cases = len(results)
    print(f"Benchmarks have run for the following {number_of_cases:02} cases: ")

    cases = list(results.keys())
    for count_case in range(number_of_cases):
        print(f"Case {count_case:02}: {cases[count_case]}")

    summary = defaultdict(dict)
    for count_case in range(number_of_cases):

        print(f"\n------ Summary of Case {count_case:02}: {cases[count_case]}: ")
        parameter = []
        time_min = []
        time_max = []
        time_avg = []
        benchmark_pairs = results[cases[count_case]]
        number_of_benchmarks = len(benchmark_pairs)
        print(f"Number of benchmarks = {number_of_benchmarks:04} ")

        for parameter_tuple in benchmark_pairs.keys():
            parameter.append(parameter_tuple[0])
            times = benchmark_pairs[parameter_tuple]["time"]
            time_min.append(min(times))
            time_max.append(max(times))
            time_avg.append(sum(times) / len(times))

        summary[cases[count_case]]["parameter"] = parameter
        summary[cases[count_case]]["time_min"] = time_min
        summary[cases[count_case]]["time_max"] = time_max
        summary[cases[count_case]]["time_avg"] = time_avg

        for count_benchmark in range(number_of_benchmarks):
            smry = summary[cases[count_case]]
            print(
                f"par: {smry['parameter'][count_benchmark]} {smry['time_min'][count_benchmark]:>#7.2g}s (min) {smry['time_max'][count_benchmark]:>#7.2g}s (max) {smry['time_avg'][count_benchmark]:>#7.2g}s (avg) "
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
