"""Utilities for plotting benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _set_axis_properties(
    ax: plt.Axes,
    parameter_values: np.ndarray,
    parameter_label: str,
    measurement_label: str,
) -> None:
    ax.set(
        xlabel=parameter_label,
        ylabel=measurement_label,
        xscale="log",
        yscale="log",
        xticks=parameter_values,
        xticklabels=parameter_values,
    )
    ax.minorticks_off()


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


def plot_times(
    ax: plt.Axes, parameter_symbol: str, parameter_values: np.ndarray, results: dict
) -> None:
    min_times = np.array([min(r["run_times_in_seconds"]) for r in results])
    mid_times = np.array([np.median(r["run_times_in_seconds"]) for r in results])
    max_times = np.array([max(r["run_times_in_seconds"]) for r in results])
    ax.plot(parameter_values, mid_times, label="Measured")
    ax.fill_between(parameter_values, min_times, max_times, alpha=0.5)
    _plot_scaling_guide(ax, parameter_symbol, parameter_values, mid_times, 3)
    ax.legend()


def plot_flops(
    ax: plt.Axes, parameter_symbol: str, parameter_values: np.ndarray, results: dict
) -> None:
    flops = np.array([r["cost_analysis"]["flops"] for r in results])
    ax.plot(parameter_values, flops, label="Measured")
    _plot_scaling_guide(ax, parameter_symbol, parameter_values, flops, 2)
    ax.legend()


def plot_error(
    ax: plt.Axes, parameter_symbol: str, parameter_values: np.ndarray, results: dict
) -> None:
    max_abs_errors = np.array([r["max_abs_error"] for r in results])
    mean_abs_errors = np.array([r["mean_abs_error"] for r in results])
    ax.plot(parameter_values, max_abs_errors, label="max(abs(error))")
    ax.plot(parameter_values, mean_abs_errors, label="mean(abs(error))")
    _plot_scaling_guide(
        ax,
        parameter_symbol,
        parameter_values,
        (max_abs_errors + mean_abs_errors) / 2,
        2,
    )
    ax.legend()


def plot_memory(
    ax: plt.Axes, parameter_symbol: str, parameter_values: np.ndarray, results: dict
) -> None:
    bytes_accessed = np.array([r["cost_analysis"]["bytes_accessed"] for r in results])
    temp_size_in_bytes = np.array(
        [r["memory_analysis"]["temp_size_in_bytes"] for r in results]
    )
    output_size_in_bytes = np.array(
        [r["memory_analysis"]["output_size_in_bytes"] for r in results]
    )
    generated_code_size_in_bytes = np.array(
        [r["memory_analysis"]["generated_code_size_in_bytes"] for r in results]
    )
    ax.plot(parameter_values, bytes_accessed, label="Accesses")
    ax.plot(parameter_values, temp_size_in_bytes, label="Temporary allocations")
    ax.plot(parameter_values, output_size_in_bytes, label="Output size")
    ax.plot(parameter_values, generated_code_size_in_bytes, label="Generated code size")
    _plot_scaling_guide(
        ax,
        parameter_symbol,
        parameter_values,
        (bytes_accessed + output_size_in_bytes) / 2,
        2,
    )
    ax.legend()


_measurement_plot_functions_and_labels = {
    "times": (plot_times, "Run time / s"),
    "flops": (plot_flops, "Floating point operations"),
    "memory": (plot_memory, "Memory / B"),
    "error": (plot_error, "Numerical error"),
}


def plot_results_against_bandlimit(
    benchmark_results_path: str | Path,
    functions: tuple[str] = ("forward", "inverse"),
    measurements: tuple[str] = ("times", "flops", "memory", "error"),
    axis_size: float = 3.0,
    fig_dpi: int = 100,
    functions_along_columns: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    benchmark_results_path = Path(benchmark_results_path)
    with benchmark_results_path.open("r") as f:
        benchmark_results = json.load(f)
    n_functions = len(functions)
    n_measurements = len(measurements)
    n_rows, n_cols = (
        (n_measurements, n_functions)
        if functions_along_columns
        else (n_functions, n_measurements)
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(axis_size * n_cols, axis_size * n_rows),
        dpi=fig_dpi,
        squeeze=False,
    )
    axes = axes.T if functions_along_columns else axes
    for axes_row, function in zip(axes, functions, strict=False):
        results = benchmark_results["results"][function]
        l_values = np.array([r["parameters"]["L"] for r in results])
        for ax, measurement in zip(axes_row, measurements, strict=False):
            plot_function, label = _measurement_plot_functions_and_labels[measurement]
            try:
                plot_function(ax, "L", l_values, results)
                ax.set(title=function)
            except KeyError:
                ax.axis("off")
            _set_axis_properties(ax, l_values, "Bandlimit $L$", label)
    return fig, ax


def _parse_cli_arguments() -> argparse.Namespace:
    """Parse rguments passed for plotting command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate plot from benchmark results file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-results-path",
        type=Path,
        help="Path to JSON file containing benchmark results to plot.",
    )
    parser.add_argument(
        "-output-path",
        type=Path,
        help="Path to write figure to.",
    )
    parser.add_argument(
        "-functions",
        nargs="+",
        help="Names of functions to plot. forward and inverse are plotted if omitted.",
    )
    parser.add_argument(
        "-measurements",
        nargs="+",
        help="Names of measurements to plot. All functions are plotted if omitted.",
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
    functions = (
        ("forward", "inverse") if args.functions is None else tuple(args.functions)
    )
    measurements = (
        ("times", "flops", "memory", "error")
        if args.measurements is None
        else tuple(args.measurements)
    )
    fig, _ = plot_results_against_bandlimit(
        args.results_path,
        functions=functions,
        measurements=measurements,
        axis_size=args.axis_size,
        fig_dpi=args.dpi,
        functions_along_columns=args.functions_along_columns,
    )
    if args.title is not None:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output_path)
