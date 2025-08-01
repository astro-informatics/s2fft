"""Utilities for plotting benchmark results."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np


class Metric(NamedTuple):
    label: str
    extract_metric: Callable[[dict], float]
    expected_scaling_order: int | None = (None,)
    multiple_measurements: bool = False


def _set_axis_properties(
    ax: plt.Axes,
    parameter_values: np.ndarray,
    parameter_label: str,
    measurement_label: str,
    function_label: str,
) -> None:
    ax.set(
        xlabel=parameter_label,
        ylabel=measurement_label,
        xscale="log",
        yscale="log",
        xticks=parameter_values,
        xticklabels=parameter_values,
        title=function_label,
    )
    ax.xaxis.minorticks_off()


def _plot_scaling_guide(
    ax: plt.Axes,
    parameter_symbol: str,
    parameter_values: np.ndarray,
    measurement_values: np.ndarray,
    order: int,
) -> None:
    n = np.argsort(parameter_values)[len(parameter_values) // 2]
    coefficient = measurement_values[n] / float(parameter_values[n]) ** order
    ax.plot(
        parameter_values,
        coefficient * parameter_values.astype(float) ** order,
        "k:",
        label=f"$\\mathcal{{O}}({parameter_symbol}^{order})$",
    )


def _extract_and_plot_metric_values(
    ax: plt.Axes,
    metric: Metric,
    results: dict,
    parameter_values: np.ndarray,
    results_label: str | None = None,
) -> np.ndarray:
    if metric.multiple_measurements:
        metric_values = {
            "min": np.array([min(metric.extract_metric(r)) for r in results]),
            "median": np.array([np.median(metric.extract_metric(r)) for r in results]),
            "max": np.array([max(metric.extract_metric(r)) for r in results]),
        }
        ax.plot(
            parameter_values,
            metric_values["median"],
            label=results_label,
        )
        ax.fill_between(
            parameter_values,
            metric_values["min"],
            metric_values["max"],
            alpha=0.5,
        )
        return metric_values["median"]
    else:
        metric_values = np.array([metric.extract_metric(r) for r in results])
        ax.plot(parameter_values, metric_values, label=results_label)
        return metric_values


_metrics = {
    "run_times": Metric("Run time / s", lambda r: r["run_times_in_seconds"], 3, True),
    "compilation_times": Metric(
        "Compilation time / s",
        lambda r: r["compilation_time_in_seconds"],
        None,
    ),
    "flops": Metric(
        "Floating point operations", lambda r: r["cost_analysis"]["flops"], 2
    ),
    "memory_accesses": Metric(
        "Memory accesses / B", lambda r: r["cost_analysis"]["bytes_accessed"], 2
    ),
    "memory_allocations": Metric(
        "Memory allocations / B",
        lambda r: r["memory_analysis"]["temp_size_in_bytes"],
        2,
    ),
    "argument_size": Metric(
        "Argument size / B", lambda r: r["memory_analysis"]["argument_size_in_bytes"], 2
    ),
    "output_size": Metric(
        "Output size / B", lambda r: r["memory_analysis"]["output_size_in_bytes"], 2
    ),
    "traced_memory_peak": Metric(
        "Traced memory peak / B", lambda r: r["traced_memory_peak_in_bytes"], 3
    ),
    "code_size": Metric(
        "Generated code size / B",
        lambda r: r["memory_analysis"]["generated_code_size_in_bytes"],
        None,
    ),
    "mean_abs_error": Metric(
        "Numerical error (mean)", lambda r: r["mean_abs_error"], None
    ),
    "max_abs_error": Metric(
        "Numerical error (max)", lambda r: r["max_abs_error"], None
    ),
}


def plot_results_against_parameter(
    benchmark_results_paths: list[str | Path],
    functions: tuple[str],
    metric_names: tuple[str],
    parameter_name: str,
    parameter_label: str,
    axis_size: float = 5.0,
    fig_dpi: int = 100,
    functions_along_columns: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    n_functions = len(functions)
    n_metrics = len(metric_names)
    n_rows, n_cols = (
        (n_metrics, n_functions)
        if functions_along_columns
        else (n_functions, n_metrics)
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(axis_size * n_cols, axis_size * n_rows),
        dpi=fig_dpi,
        squeeze=False,
    )
    axes = axes.T if functions_along_columns else axes
    for results_index, benchmark_results_path in enumerate(benchmark_results_paths):
        last_results = results_index == len(benchmark_results_paths) - 1
        benchmark_results_path = Path(benchmark_results_path)
        with benchmark_results_path.open("r") as f:
            benchmark_results = json.load(f)
        for axes_row, function in zip(axes, functions, strict=False):
            results = sorted(
                benchmark_results["results"][function],
                key=lambda r: r["parameters"][parameter_name],
            )
            parameter_values = np.array(
                [r["parameters"][parameter_name] for r in results]
            )
            if len(set(parameter_values)) != len(parameter_values):
                raise ValueError(
                    f"Non unique parameter values detected for function {function} "
                    f"in file {benchmark_results_path}."
                )
            for ax, metric_name in zip(axes_row, metric_names, strict=False):
                metric = _metrics[metric_name]
                try:
                    metric_values = _extract_and_plot_metric_values(
                        ax,
                        metric,
                        results,
                        parameter_values,
                        benchmark_results_path.stem,
                    )
                    if last_results and metric.expected_scaling_order is not None:
                        _plot_scaling_guide(
                            ax,
                            parameter_name,
                            parameter_values,
                            metric_values,
                            metric.expected_scaling_order,
                        )
                    ax.legend()
                    _set_axis_properties(
                        ax, parameter_values, parameter_label, metric.label, function
                    )
                except KeyError:
                    ax.axis("off")
    return fig, ax


def _parse_cli_arguments() -> argparse.Namespace:
    """Parse arguments passed for plotting command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate plot from benchmark results file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-results-path",
        type=Path,
        nargs="+",
        help="Path(s) to JSON file containing benchmark results to plot.",
    )
    parser.add_argument(
        "-output-path",
        type=Path,
        help="Path to write figure to.",
    )
    parser.add_argument(
        "-functions",
        nargs="+",
        help="Names of functions to plot.",
        metavar="FUNCTION",
        default=["forward", "inverse"],
    )
    parser.add_argument(
        "-metrics",
        nargs="+",
        help="Names of metrics to plot.",
        metavar="METRIC",
        default=[
            "run_times",
            "compilation_times",
            "memory_allocations",
        ],
    )
    parser.add_argument(
        "-parameter",
        nargs=2,
        help="Key and label for parameter to plot metrics against.",
        metavar=("NAME", "LABEL"),
        default=["L", "Bandlimit $L$"],
    )
    parser.add_argument(
        "-axis-size", type=float, default=5.0, help="Size of each plot axis in inches."
    )
    parser.add_argument(
        "-dpi", type=int, default=100, help="Figure resolution in dots per inch."
    )
    parser.add_argument(
        "-title", type=str, help="Title for figure. No title added if omitted."
    )
    parser.add_argument(
        "--functions-along-columns",
        action="store_true",
        help="Whether to orient axes with functions along columns instead of rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_arguments()
    parameter_name, parameter_label = args.parameter
    fig, _ = plot_results_against_parameter(
        args.results_path,
        functions=args.functions,
        metric_names=args.metrics,
        parameter_name=parameter_name,
        parameter_label=parameter_label,
        axis_size=args.axis_size,
        fig_dpi=args.dpi,
        functions_along_columns=args.functions_along_columns,
    )
    if args.title is not None:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output_path)
