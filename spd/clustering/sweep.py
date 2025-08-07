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

from spd.clustering.merge import MergeConfig, MergeHistory, merge_iteration


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
        # TODO: adapt to new rank cost func
        for act_thresh, check_thresh, alpha, _rank_func in itertools.product(
            self.activation_thresholds, self.check_thresholds, self.alphas, self.rank_cost_funcs
        ):
            merge_config = MergeConfig(
                activation_threshold=act_thresh,
                alpha=alpha,
                check_threshold=check_thresh,
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
        cmap = cm.viridis  # pyright: ignore[reportAttributeAccessIssue]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return norm, {"cmap": cmap, "sm": sm}
    else:
        colors: np.ndarray = cm.viridis(np.linspace(0, 1, len(line_values)))  # pyright: ignore[reportAttributeAccessIssue]
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
    raw_activations: torch.Tensor,
    sweep_config: SweepConfig,
    component_labels: list[str],
) -> list[MergeHistory]:
    """Run hyperparameter sweep across all parameter combinations."""
    configs = sweep_config.generate_configs()
    print(f"{len(configs) = }")

    results: list[MergeHistory] = []

    for _i, merge_config in tqdm(enumerate(configs), total=len(configs)):
        try:
            merge_history = merge_iteration(
                activations=raw_activations,
                merge_config=merge_config,
                component_labels=component_labels,
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