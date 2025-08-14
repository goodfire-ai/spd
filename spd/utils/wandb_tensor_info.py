"""Minimal WandB tensor logging utilities using muutils."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
from muutils.tensor_info import array_info
from torch import Tensor

from spd.log import logger


def _create_histogram(info: dict[str, Any], tensor: Tensor, name: str) -> plt.Figure:
    """Create histogram with stats markers."""
    if info["status"] != "ok" or info["size"] == 0:
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"{info['status']}", ha="center", va="center")
        ax.set_title(f"{name} - {info['status']}")
        return fig

    # Get values for histogram
    values: np.ndarray = tensor.flatten().detach().cpu().numpy()
    if info["has_nans"]:
        values = values[~np.isnan(values)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add stat lines
    if info["mean"] is not None:
        mean_val: float = info["mean"]
        median_val: float = info["median"]
        std_val: float = info["std"]

        ax.axvline(
            mean_val,
            color="red",
            linestyle="-",
            linewidth=2,
            label="$\\mu$",
        )
        ax.axvline(
            median_val,
            color="blue",
            linestyle="-",
            linewidth=2,
            label="$\\tilde{x}$",
        )
        if std_val:
            ax.axvline(
                mean_val + std_val,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="$\\mu+\\sigma$",
            )
            ax.axvline(
                mean_val - std_val,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="$\\mu-\\sigma$",
            )

    # Build informative title with tensor stats
    shape_str: str = str(tuple(info["shape"])) if "shape" in info else "unknown"
    dtype_str: str = str(info.get("dtype", "unknown")).replace("torch.", "")

    title_line1: str = f"{name}"
    title_line2: str = f"shape={shape_str}, dtype={dtype_str}"
    title_line3: str = (
        f"range=[{info['min']:.3g}, {info['max']:.3g}], "
        f"$\\mu$={mean_val:.3g}, $\\tilde{{x}}$={median_val:.3g}, $\\sigma$={std_val:.3g}"
    )

    # Combine into multi-line title
    full_title: str = f"{title_line1}\n{title_line2}\n{title_line3}"
    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def wandb_log_tensor(
    run: wandb.sdk.wandb_run.Run,
    data: Tensor | dict[str, Tensor],
    name: str,
    step: int,
) -> None:
    """Log tensor(s) with stats to WandB as metrics and histograms.

    Args:
        run: Current WandB run (required)
        data: Either a Tensor or dict[str, Tensor]
        name: Name for logging
        step: WandB step
    """
    if isinstance(data, dict):
        # Handle dict of tensors
        for key, tensor in data.items():
            full_name: str = f"{name}.{key}"
            _log_one(run, tensor, full_name, step)
    else:
        # Handle single tensor
        _log_one(run, data, name, step)


def _log_one(
    run: wandb.sdk.wandb_run.Run,
    tensor: Tensor,
    name: str,
    step: int,
) -> None:
    """Log a single tensor."""
    # Get tensor info once
    info: dict[str, Any] = array_info(tensor)

    # Create and log histogram
    fig: plt.Figure = _create_histogram(info, tensor, name)
    histogram_key: str = f"tensor_histograms/{name}"
    run.log({histogram_key: fig}, step=step)
    plt.close(fig)

    # Log numeric stats as metrics (viewable like loss) using dict comprehension
    stats_to_log: dict[str, float] = {
        f"tensor_metrics/{name}/{key}": info[key]
        for key in ["mean", "std", "median", "min", "max"]
        if key in info and info[key] is not None
    }

    # Add nan_percent if present
    nan_percent: float = info.get("nan_percent", 0)
    if nan_percent > 0:
        stats_to_log[f"tensor_metrics/{name}/nan_percent"] = nan_percent

    if stats_to_log:
        run.log(stats_to_log, step=step)

    # Log links using the SPD logger
    run_url: str | None = run.get_url()
    if run_url:
        # Build full URLs for histogram and metrics
        histogram_url: str = f"{run_url}#custom-charts/tensor_histograms/{name.replace('/', '%2F')}"
        metrics_url: str = f"{run_url}#scalars/section=tensor_metrics%2F{name.replace('/', '%2F')}"

        logger.info(
            f"Logged tensor: {name}\n  Histogram: {histogram_url}\n  Metrics: {metrics_url}"
        )
