import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from tqdm import tqdm

from spd.clustering.merge import MergeConfig, MergeHistory, MergePlotConfig, merge_iteration


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""

    activation_thresholds: list[float]
    check_thresholds: list[float]
    alphas: list[float]
    rank_cost_funcs: list[Callable[[float], float]]
    iters: int = 100

    def generate_configs(self) -> list[MergeConfig]:
        """Generate all MergeConfig combinations."""
        configs = []
        for act_thresh, check_thresh, alpha, rank_func in itertools.product(
            self.activation_thresholds, self.check_thresholds, self.alphas, self.rank_cost_funcs
        ):
            merge_config = MergeConfig(
                activation_threshold=act_thresh,
                alpha=alpha,
                check_threshold=check_thresh,
                rank_cost_fn=rank_func,
                iters=self.iters,
            )
            configs.append(merge_config)
        return configs


def format_value(val: Any) -> str:
    """Format value to max 3 digits precision."""
    if isinstance(val, float):
        return f"{val:.3g}"
    return str(val)


def format_range(values: list[Any]) -> str:
    """Format range of values."""
    if len(values) == 1:
        return format_value(values[0])
    elif len(values) == 2:
        return f"[{format_value(values[0])}, {format_value(values[-1])}]"
    else:
        return f"[{format_value(values[0])}...{format_value(values[-1])}]"


def get_unique_param_values(results: list[MergeHistory]) -> dict[str, list[Any]]:
    """Extract unique parameter values from results."""
    all_params: list[str] = ["activation_threshold", "check_threshold", "alpha", "rank_cost_name"]
    return {
        param: sorted(list(set(r.sweep_params[param] for r in results if r.sweep_params)))
        for param in all_params
    }


def filter_results_by_params(
    results: list[MergeHistory], fixed_params: dict[str, Any]
) -> list[MergeHistory]:
    """Filter results by fixed parameter values."""
    filtered_results: list[MergeHistory] = results
    for param, value in fixed_params.items():
        filtered_results = [
            r for r in filtered_results if r.sweep_params and r.sweep_params[param] == value
        ]
    return filtered_results


def validate_plot_params(
    lines_by: str, rows_by: str, cols_by: str, fixed_params: dict[str, Any]
) -> None:
    """Validate that all required parameters are fixed for 3D plotting."""
    all_params: list[str] = ["activation_threshold", "check_threshold", "alpha", "rank_cost_name"]
    used_params: set[str] = {lines_by, rows_by, cols_by}
    unused_params: list[str] = [p for p in all_params if p not in used_params]

    missing_fixed: list[str] = [p for p in unused_params if p not in fixed_params]
    if missing_fixed:
        raise ValueError(f"Must fix all unused parameters. Missing: {missing_fixed}")


def create_colormap(line_values: list[Any]) -> tuple[Any, Any]:
    """Create colormap for line parameter."""
    if isinstance(line_values[0], int | float):
        norm = LogNorm(vmin=min(line_values), vmax=max(line_values))
        cmap = cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return norm, {"cmap": cmap, "sm": sm}
    else:
        colors: np.ndarray = cm.viridis(np.linspace(0, 1, len(line_values)))
        color_dict: dict[Any, Any] = {val: colors[i] for i, val in enumerate(line_values)}
        return None, {"color_dict": color_dict}


def create_suptitle(
    lines_by: str,
    rows_by: str,
    cols_by: str,
    param_values: dict[str, list[Any]],
    fixed_params: dict[str, Any],
) -> str:
    """Create informative suptitle showing parameter organization."""
    title_parts: list[str] = []
    title_parts.append(f"lines_by: {lines_by} ∈ {format_range(param_values[lines_by])}")
    title_parts.append(f"rows_by: {rows_by} ∈ {format_range(param_values[rows_by])}")
    title_parts.append(f"cols_by: {cols_by} ∈ {format_range(param_values[cols_by])}")

    if fixed_params:
        fixed_str: str = ", ".join([f"{k}={format_value(v)}" for k, v in fixed_params.items()])
        title_parts.append(f"fixed values: {fixed_str}")

    return "\n".join(title_parts)


def process_values(values: np.ndarray, normalize_to_zero: bool, log_delta: bool) -> np.ndarray:
    """Process metric values with normalization and log transformation."""
    if normalize_to_zero and len(values) > 0:
        values = values - values[0]
        if log_delta and len(values) > 1:
            values = np.sign(values) * np.log10(np.abs(values) + 1e-10)
    return values


def get_iterations(n_values: int, log_iterations: bool) -> np.ndarray:
    """Get iteration values, optionally with log scaling."""
    iterations: np.ndarray = np.array(range(n_values))
    if log_iterations:
        iterations = iterations + 1  # Start from 1 for log scale
    return iterations


def setup_axis(
    ax: plt.Axes,
    row_idx: int,
    col_idx: int,
    n_rows: int,
    n_cols: int,
    row_val: Any,
    col_val: Any,
    rows_by: str,
    cols_by: str,
    metric: str,
    normalize_to_zero: bool,
    log_delta: bool,
    log_iterations: bool,
) -> None:
    """Set up axis labels and scales."""
    if row_idx == 0:  # Only first row gets titles
        ax.set_title(f"{cols_by}\n{format_value(col_val)}")

    if log_iterations:
        ax.set_xscale("log")

    if row_idx == n_rows - 1:  # Only bottom row gets x-labels
        xlabel: str = "log(Iteration)" if log_iterations else "Iteration"
        ax.set_xlabel(xlabel)

    if col_idx == 0:  # Only leftmost column gets y-label
        ylabel: str = metric.split("_")[-1].title()
        if normalize_to_zero:
            ylabel = f"log|Δ{ylabel}|" if log_delta else f"Δ{ylabel}"
        ax.set_ylabel(ylabel)

    if col_idx == n_cols - 1:  # Rightmost column gets row value
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"{rows_by}\n={format_value(row_val)}", rotation=270, labelpad=25)

    ax.grid(True, alpha=0.3)


def add_colorbar_or_legend(
    fig: plt.Figure,
    axes: np.ndarray,
    line_values: list[Any],
    lines_by: str,
    colormap_info: dict[str, Any],
) -> None:
    """Add colorbar for numeric parameters or legend for categorical."""
    if isinstance(line_values[0], int | float):
        cbar = fig.colorbar(
            colormap_info["sm"], ax=axes, orientation="horizontal", fraction=0.05, pad=-0.25
        )
        cbar.set_label(lines_by)
    else:
        color_dict: dict[Any, Any] = colormap_info["color_dict"]
        legend_elements: list[Line2D] = [
            Line2D([0], [0], color=color_dict[val], lw=2, label=f"{lines_by}={format_value(val)}")
            for val in line_values
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(line_values),
        )


def run_hyperparameter_sweep(
    raw_activations: torch.Tensor, sweep_config: SweepConfig
) -> list[MergeHistory]:
    """Run hyperparameter sweep across all parameter combinations."""
    configs = sweep_config.generate_configs()
    print(f"{len(configs) = }")

    results: list[MergeHistory] = []

    for _i, merge_config in tqdm(enumerate(configs), total=len(configs)):
        try:
            plot_config = MergePlotConfig(
                plot_every=0,  # No plotting during sweep
                plot_final=False,
            )

            merge_history, final_merge = merge_iteration(
                activations=raw_activations,
                merge_config=merge_config,
                component_labels=None,
                plot_config=plot_config,
            )

            # Store sweep parameters in the merge history for later use
            merge_history.sweep_params = {
                "activation_threshold": merge_config.activation_threshold,
                "check_threshold": merge_config.check_threshold,
                "alpha": merge_config.alpha,
                "rank_cost_name": merge_config.rank_cost_fn.__name__,
            }
            results.append(merge_history)
        except Exception as e:
            print(f"Failed: {e}")

    print(f"{len(results) = }")
    return results


def plot_evolution_histories(
    results: list[MergeHistory],
    fixed_params: dict[str, Any],
    metric: str = "non_diag_costs_min",
    lines_by: str = "alpha",
    rows_by: str = "activation_threshold",
    cols_by: str = "check_threshold",
    figsize: tuple[int, int] = (15, 10),
    normalize_to_zero: bool = True,
    log_delta: bool = True,
    log_iterations: bool = False,
) -> None:
    """Plot evolution histories with 3D parameter organization."""

    validate_plot_params(lines_by, rows_by, cols_by, fixed_params)

    filtered_results: list[MergeHistory] = filter_results_by_params(results, fixed_params)
    if not filtered_results:
        raise ValueError(f"No results match fixed parameters: {fixed_params}")

    param_values: dict[str, list[Any]] = get_unique_param_values(filtered_results)
    row_values: list[Any] = param_values[rows_by]
    col_values: list[Any] = param_values[cols_by]
    line_values: list[Any] = param_values[lines_by]

    n_rows: int = len(row_values)
    n_cols: int = len(col_values)

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True)

    norm, colormap_info = create_colormap(line_values)

    suptitle: str = create_suptitle(lines_by, rows_by, cols_by, param_values, fixed_params)
    fig.suptitle(suptitle, fontsize=12)

    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax: plt.Axes = axes[row_idx, col_idx]

            subset_results: list[MergeHistory] = [
                r
                for r in filtered_results
                if r.sweep_params
                and r.sweep_params[rows_by] == row_val
                and r.sweep_params[cols_by] == col_val
            ]

            for line_val in line_values:
                line_results: list[MergeHistory] = [
                    r
                    for r in subset_results
                    if r.sweep_params and r.sweep_params[lines_by] == line_val
                ]

                if line_results:
                    result: MergeHistory = line_results[0]
                    values: np.ndarray = np.array(getattr(result, metric))
                    values = process_values(values, normalize_to_zero, log_delta)
                    iterations: np.ndarray = get_iterations(len(values), log_iterations)

                    if isinstance(line_values[0], int | float):
                        color = colormap_info["cmap"](norm(line_val))
                    else:
                        color = colormap_info["color_dict"][line_val]

                    ax.plot(iterations, values, color=color, alpha=0.8, linewidth=2)

            setup_axis(
                ax,
                row_idx,
                col_idx,
                n_rows,
                n_cols,
                row_val,
                col_val,
                rows_by,
                cols_by,
                metric,
                normalize_to_zero,
                log_delta,
                log_iterations,
            )

    add_colorbar_or_legend(fig, axes, line_values, lines_by, colormap_info)
    plt.tight_layout()
    plt.show()


def detect_most_variable_params(results: list[MergeHistory]) -> tuple[str, str]:
    """Detect which parameters have the most variation for smart defaults."""
    all_params: list[str] = ["activation_threshold", "check_threshold", "alpha", "rank_cost_name"]
    param_values: dict[str, list[Any]] = get_unique_param_values(results)

    # Count unique values for each parameter
    param_counts: dict[str, int] = {param: len(param_values[param]) for param in all_params}

    # Sort by number of unique values (most variable first)
    sorted_params: list[tuple[str, int]] = sorted(
        param_counts.items(), key=lambda x: x[1], reverse=True
    )

    # Return top 2 most variable parameters
    if len(sorted_params) >= 2:
        return sorted_params[0][0], sorted_params[1][0]
    else:
        return "activation_threshold", "check_threshold"


def create_multiple_heatmaps(
    results: list[MergeHistory],
    statistics: list[tuple[Callable[[MergeHistory], float], str]],
    fixed_params: dict[str, Any],
    x_by: str = "check_threshold",
    y_by: str = "activation_threshold",
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Create multiple heatmaps with same parameters but different statistics.

    Args:
        results: List of sweep results
        statistics: List of (statistic_func, statistic_name) tuples
        fixed_params: Parameters to fix
        x_by: Parameter for x-axis
        y_by: Parameter for y-axis
        **kwargs: Additional arguments passed to create_heatmaps

    Returns:
        Dictionary mapping statistic names to heatmap results
    """
    heatmap_results: dict[str, dict[str, Any]] = {}

    for stat_func, stat_name in statistics:
        heatmap_results[stat_name] = create_heatmaps(
            results=results,
            fixed_params=fixed_params,
            statistic_func=stat_func,
            statistic_name=stat_name,
            x_by=x_by,
            y_by=y_by,
            **kwargs,
        )

    return heatmap_results


def create_heatmaps(
    results: list[MergeHistory],
    fixed_params: dict[str, Any],
    statistic_func: Callable[[MergeHistory], float],
    statistic_name: str,
    x_by: str = "check_threshold",
    y_by: str = "activation_threshold",
    aggregation: str = "mean",
    log_scale: bool = False,
    normalize: bool = False,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (12, 8),
    show_sample_counts: bool = False,
) -> dict[str, Any]:
    """Create flexible heatmaps showing statistics across hyperparameter combinations.

    Args:
        results: List of sweep results
        fixed_params: Parameters to fix (remaining 2 will be heatmap axes)
        statistic_func: Function to extract statistic from SweepResult
        statistic_name: Name for plot titles and labels
        x_by: Parameter for x-axis
        y_by: Parameter for y-axis
        aggregation: How to aggregate multiple values ('mean', 'median', 'min', 'max', 'std')
        log_scale: Apply log transform to statistic values
        normalize: Normalize values to [0,1] range
        cmap: Colormap name
        figsize: Figure size
        show_sample_counts: Overlay sample count text on cells

    Returns:
        Dictionary with heatmap data and metadata
    """

    # Validate parameters
    all_params: list[str] = ["activation_threshold", "check_threshold", "alpha", "rank_cost_name"]
    used_params: set[str] = {x_by, y_by}

    if len(used_params) != 2:
        raise ValueError(f"x_by and y_by must be different. Got: {x_by}, {y_by}")

    if not used_params.issubset(all_params):
        raise ValueError(f"x_by and y_by must be from {all_params}")

    # Filter results by fixed parameters
    filtered_results: list[MergeHistory] = filter_results_by_params(results, fixed_params)
    if not filtered_results:
        raise ValueError(f"No results match fixed parameters: {fixed_params}")

    # Get unique values for heatmap axes
    param_values: dict[str, list[Any]] = get_unique_param_values(filtered_results)
    x_values: list[Any] = param_values[x_by]
    y_values: list[Any] = param_values[y_by]

    # Create heatmap data
    heatmap_data: np.ndarray = np.full((len(y_values), len(x_values)), np.nan)
    sample_counts: np.ndarray = np.zeros((len(y_values), len(x_values)), dtype=int)

    for y_idx, y_val in enumerate(y_values):
        for x_idx, x_val in enumerate(x_values):
            matching_results: list[MergeHistory] = [
                r
                for r in filtered_results
                if r.sweep_params
                and r.sweep_params[x_by] == x_val
                and r.sweep_params[y_by] == y_val
            ]

            if matching_results:
                stat_values: list[float] = [statistic_func(r) for r in matching_results]
                sample_counts[y_idx, x_idx] = len(stat_values)

                if aggregation == "mean":
                    heatmap_data[y_idx, x_idx] = np.mean(stat_values)
                elif aggregation == "median":
                    heatmap_data[y_idx, x_idx] = np.median(stat_values)
                elif aggregation == "min":
                    heatmap_data[y_idx, x_idx] = np.min(stat_values)
                elif aggregation == "max":
                    heatmap_data[y_idx, x_idx] = np.max(stat_values)
                elif aggregation == "std":
                    heatmap_data[y_idx, x_idx] = np.std(stat_values)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

    # Apply transformations
    plot_data: np.ndarray = heatmap_data.copy()

    if log_scale:
        # Handle negative/zero values for log scale
        min_positive: float = (
            np.nanmin(plot_data[plot_data > 0]) if np.any(plot_data > 0) else 1e-10
        )
        plot_data = np.where(plot_data > 0, plot_data, min_positive)
        plot_data = np.log10(plot_data)

    if normalize:
        min_val: float = np.nanmin(plot_data)
        max_val: float = np.nanmax(plot_data)
        if max_val > min_val:
            plot_data = (plot_data - min_val) / (max_val - min_val)

    # Create plot
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(plot_data, aspect="auto", cmap=cmap, origin="lower")

    # Set ticks and labels
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([format_value(x) for x in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([format_value(y) for y in y_values])

    ax.set_xlabel(x_by.replace("_", " ").title())
    ax.set_ylabel(y_by.replace("_", " ").title())

    # Create title
    title_parts: list[str] = [statistic_name]
    if aggregation != "mean":
        title_parts.append(f"({aggregation})")
    if log_scale:
        title_parts.append("(log scale)")
    if normalize:
        title_parts.append("(normalized)")

    if fixed_params:
        fixed_str: str = ", ".join([f"{k}={format_value(v)}" for k, v in fixed_params.items()])
        title_parts.append(f"\nFixed: {fixed_str}")

    ax.set_title(" ".join(title_parts))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar_label: str = statistic_name
    if log_scale:
        cbar_label = f"log₁₀({cbar_label})"
    cbar.set_label(cbar_label)

    # Optionally show sample counts
    if show_sample_counts:
        for y_idx in range(len(y_values)):
            for x_idx in range(len(x_values)):
                count: int = sample_counts[y_idx, x_idx]
                if count > 0:
                    ax.text(
                        x_idx,
                        y_idx,
                        str(count),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                        weight="bold",
                    )

    plt.tight_layout()
    plt.show()

    return {
        "heatmap_data": heatmap_data,
        "plot_data": plot_data,
        "sample_counts": sample_counts,
        "x_values": x_values,
        "y_values": y_values,
        "x_by": x_by,
        "y_by": y_by,
        "fixed_params": fixed_params,
        "statistic_name": statistic_name,
        "aggregation": aggregation,
    }


# Example statistic functions for common analyses
def get_convergence_rate(result: MergeHistory) -> float:
    """Calculate convergence rate as slope of cost evolution."""
    costs: list[float] = result.non_diag_costs_min
    if len(costs) < 2:
        return 0.0

    # Use simple linear regression to get slope
    x: np.ndarray = np.arange(len(costs))
    y: np.ndarray = np.array(costs)

    if np.std(x) == 0:
        return 0.0

    slope: float = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
    return slope


def get_cost_reduction_ratio(result: MergeHistory) -> float:
    """Calculate ratio of final cost to initial cost."""
    costs: list[float] = result.non_diag_costs_min
    if len(costs) < 2:
        return 1.0

    initial_cost: float = costs[0]
    final_cost: float = costs[-1]

    if initial_cost == 0:
        return 1.0

    return final_cost / initial_cost


def get_merge_efficiency(result: MergeHistory) -> float:
    """Calculate groups merged per iteration."""
    if result.total_iterations == 0:
        return 0.0

    # Use the actual initial groups from the merge history
    groups_merged: int = result.initial_k_groups - result.final_k_groups
    return groups_merged / result.total_iterations


def get_early_convergence(threshold_ratio: float = 0.1) -> Callable[[MergeHistory], float]:
    """Create function to detect early convergence iterations."""

    def _early_convergence(result: MergeHistory) -> float:
        costs: list[float] = result.non_diag_costs_min
        if len(costs) < 3:
            return len(costs)

        initial_cost: float = costs[0]

        for i, cost in enumerate(costs[1:], 1):
            if initial_cost > 0 and (cost / initial_cost) <= threshold_ratio:
                return i

        return len(costs)

    return _early_convergence


def get_cost_variance(result: MergeHistory) -> float:
    """Calculate variance in selected pair costs (measure of stability)."""
    costs: list[float] = result.selected_pair_cost
    if len(costs) < 2:
        return 0.0

    return float(np.var(costs))


# Common statistic collections for easy use
BASIC_STATISTICS = [
    (lambda r: r.final_k_groups, "Final Groups"),
    (lambda r: r.total_iterations, "Total Iterations"),
    (get_cost_reduction_ratio, "Cost Reduction Ratio"),
    (get_merge_efficiency, "Merge Efficiency"),
]

ADVANCED_STATISTICS = [
    (get_convergence_rate, "Convergence Rate"),
    (get_early_convergence(0.1), "Early Convergence (10%)"),
    (get_cost_variance, "Cost Variance"),
    (lambda r: r.non_diag_costs_min[-1] if r.non_diag_costs_min else 0, "Final Cost"),
]

ALL_STATISTICS = BASIC_STATISTICS + ADVANCED_STATISTICS


def create_smart_heatmap(
    results: list[MergeHistory],
    statistic_func: Callable[[MergeHistory], float],
    statistic_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create heatmap with smart parameter selection and defaults."""

    # Detect most variable parameters
    x_param, y_param = detect_most_variable_params(results)

    # Create fixed_params for remaining parameters
    all_params: list[str] = ["activation_threshold", "check_threshold", "alpha", "rank_cost_name"]
    remaining_params: list[str] = [p for p in all_params if p not in {x_param, y_param}]

    param_values: dict[str, list[Any]] = get_unique_param_values(results)

    # Use median/middle values for fixed parameters
    fixed_params: dict[str, Any] = {}
    for param in remaining_params:
        values: list[Any] = param_values[param]
        if isinstance(values[0], int | float):
            # Use median for numeric parameters
            fixed_params[param] = values[len(values) // 2]
        else:
            # Use first value for categorical parameters
            fixed_params[param] = values[0]

    return create_heatmaps(
        results=results,
        fixed_params=fixed_params,
        statistic_func=statistic_func,
        statistic_name=statistic_name,
        x_by=x_param,
        y_by=y_param,
        **kwargs,
    )


# Simple stopping condition examples using lambdas
def cost_ratio_condition(ratio: float, metric: str) -> Callable[[MergeHistory], bool]:
    """Create stopping condition for cost ratio."""

    def condition(history: MergeHistory) -> bool:
        costs = getattr(history, metric)
        return len(costs) >= 2 and costs[-1] >= costs[0] * ratio

    return condition


def iteration_condition(max_iters: int) -> Callable[[MergeHistory], bool]:
    """Create stopping condition for max iterations."""
    return lambda history: len(history.non_diag_costs_min) >= max_iters
