"""Plotting functions for merge visualizations."""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.math.merge_distances import DistancesArray
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.clustering.util import format_scientific_latex

DEFAULT_PLOT_CONFIG: dict[str, Any] = dict(
    figsize=(16, 10),
    tick_spacing=5,
    save_pdf=False,
    pdf_prefix="merge_iteration",
)


def plot_merge_iteration(
    current_merge: GroupMerge,
    current_coact: Float[Tensor, "k_groups k_groups"],
    costs: Float[Tensor, "k_groups k_groups"],
    pair_cost: float,
    iteration: int,
    component_labels: list[str] | None = None,
    plot_config: dict[str, Any] | None = None,
    nan_diag: bool = True,
) -> None:
    """Plot merge iteration results with merge tree, coactivations, and costs.

    Args:
            current_merge: Current merge state
            current_coact: Current coactivation matrix
            costs: Current cost matrix
            pair_cost: Cost of selected merge pair
            iteration: Current iteration number
            component_labels: Component labels for axis labeling
            plot_config: Plot configuration settings
    """
    plot_config_: dict[str, Any] = {
        **DEFAULT_PLOT_CONFIG,
        **(plot_config or {}),
    }
    axs: list[plt.Axes]
    fig, axs = plt.subplots(  # pyright: ignore[reportAssignmentType]
        1, 3, figsize=plot_config_["figsize"], sharey=True, gridspec_kw={"width_ratios": [2, 1, 1]}
    )

    # Merge plot
    current_merge.plot(ax=axs[0], show=False, component_labels=component_labels)
    axs[0].set_title("Merge")

    # Coactivations plot
    coact_min: float = current_coact.min().item()
    coact_max: float = current_coact.max().item()
    if nan_diag:
        current_coact = current_coact.clone()
        current_coact.fill_diagonal_(np.nan)
    axs[1].matshow(current_coact.cpu().numpy(), aspect="equal")
    coact_min_str: str = format_scientific_latex(coact_min)
    coact_max_str: str = format_scientific_latex(coact_max)
    axs[1].set_title(f"Coactivations\n[{coact_min_str}, {coact_max_str}]")

    # Setup ticks for coactivations
    k_groups: int = current_coact.shape[0]
    minor_ticks: list[int] = list(range(0, k_groups, plot_config_["tick_spacing"]))
    axs[1].set_yticks(minor_ticks)
    axs[1].set_xticks(minor_ticks)
    axs[1].set_xticklabels([])  # Remove x-axis tick labels but keep ticks

    # Costs plot
    costs_min: float = costs.min().item()
    costs_max: float = costs.max().item()
    if nan_diag:
        costs = costs.clone()
        costs.fill_diagonal_(np.nan)
    axs[2].matshow(costs.cpu().numpy(), aspect="equal")
    costs_min_str: str = format_scientific_latex(costs_min)
    costs_max_str: str = format_scientific_latex(costs_max)
    axs[2].set_title(f"Costs\n[{costs_min_str}, {costs_max_str}]")

    # Setup ticks for costs
    axs[2].set_yticks(minor_ticks)
    axs[2].set_xticks(minor_ticks)
    axs[2].set_xticklabels([])  # Remove x-axis tick labels but keep ticks

    fig.suptitle(f"Iteration {iteration} with cost {pair_cost:.4f}")
    plt.tight_layout()

    if plot_config_["save_pdf"]:
        fig.savefig(
            f"{plot_config_['pdf_prefix']}_iter_{iteration:03d}.pdf", bbox_inches="tight", dpi=300
        )

    plt.show()


def plot_dists_distribution(
    distances: DistancesArray,
    mode: Literal["points", "dist"] = "points",
    label: str | None = None,
    ax: plt.Axes | None = None,
    kwargs_fig: dict[str, Any] | None = None,
    kwargs_plot: dict[str, Any] | None = None,
) -> plt.Axes:
    n_iters: int = distances.shape[0]
    n_ens: int = distances.shape[1]
    assert distances.shape[2] == n_ens, "Distances must be square"

    # Ensure ax and kwargs_fig are not both provided
    if ax is not None and kwargs_fig is not None:
        raise ValueError("Cannot provide both ax and kwargs_fig")

    dists_flat: Float[np.ndarray, "n_iters n_ens*n_ens"] = distances.reshape(distances.shape[0], -1)

    # Create figure if ax not provided
    if ax is None:
        _fig, ax_ = plt.subplots(  # pyright: ignore[reportCallIssue]
            1,
            1,
            **dict(
                figsize=(8, 5),  # pyright: ignore[reportArgumentType]
                **(kwargs_fig or {}),
            ),
        )
    else:
        ax_ = ax

    if mode == "points":
        # Original points mode
        n_samples: int = dists_flat.shape[1]
        for i in range(n_iters):
            ax_.plot(
                np.full((n_samples), i),
                dists_flat[i],
                **dict(  # pyright: ignore[reportArgumentType]
                    marker="o",
                    linestyle="",
                    color="blue",
                    alpha=min(1, 10 / (n_ens * n_ens)),
                    markersize=5,
                    markeredgewidth=0,
                    **(kwargs_plot or {}),
                ),
            )
    elif mode == "dist":
        # Distribution statistics mode
        # Generate a random color for this plot
        color = np.random.rand(
            3,
        )

        # Calculate statistics for each iteration
        mins = []
        maxs = []
        means = []
        medians = []
        q1s = []
        q3s = []

        for i in range(n_iters):
            # Filter out NaN values (diagonal and upper triangle)
            valid_dists = dists_flat[i][~np.isnan(dists_flat[i])]
            if len(valid_dists) > 0:
                mins.append(np.min(valid_dists))
                maxs.append(np.max(valid_dists))
                means.append(np.mean(valid_dists))
                medians.append(np.median(valid_dists))
                q1s.append(np.percentile(valid_dists, 25))
                q3s.append(np.percentile(valid_dists, 75))
            else:
                # Handle case with no valid distances
                mins.append(np.nan)
                maxs.append(np.nan)
                means.append(np.nan)
                medians.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)

        iterations = np.arange(n_iters)

        # Plot statistics
        ax_.plot(iterations, mins, "-", color=color, alpha=0.5)
        ax_.plot(iterations, maxs, "-", color=color, alpha=0.5)
        ax_.plot(iterations, means, "-", color=color, linewidth=2, label=label)
        ax_.plot(iterations, medians, "--", color=color, linewidth=2)
        ax_.plot(iterations, q1s, ":", color=color, alpha=0.7)
        ax_.plot(iterations, q3s, ":", color=color, alpha=0.7)

        # Shade between quartiles
        ax_.fill_between(iterations, q1s, q3s, color=color, alpha=0.2)

    ax_.set_xlabel("Iteration #")
    ax_.set_ylabel("permutation invariant hamming distance")
    ax_.set_title("Distribution of pairwise distances between group merges in an ensemble")

    return ax_


def plot_merge_history_costs(
    history: MergeHistory,
    figsize: tuple[int, int] = (10, 5),
    fmt: str = "pdf",
    file_prefix: str | None = None,
) -> None:
    """Plot cost evolution from merge history"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history.max_considered_cost, label="max considered cost")
    ax.plot(history.non_diag_costs_min, label="non-diag costs min")
    ax.plot(history.non_diag_costs_max, label="non-diag costs max")
    ax.plot(history.selected_pair_cost, label="selected pair cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.legend()

    if file_prefix:
        fig.savefig(f"{file_prefix}_cost_evolution.{fmt}", bbox_inches="tight", dpi=300)


def plot_merge_history_cluster_sizes(
    history: MergeHistory,
    figsize: tuple[int, int] = (10, 5),
    fmt: str = "png",
    file_prefix: str | None = None,
) -> None:
    k_groups_t: Int[Tensor, " n_iters"] = history.k_groups
    valid_mask: Bool[Tensor, " n_iters"] = k_groups_t.ne(-1)
    has_data: bool = bool(valid_mask.any().item())
    if not has_data:
        raise ValueError("No populated iterations in history.k_groups")

    group_idxs_all: Int[Tensor, " n_iters n_components"] = history.merges.group_idxs[valid_mask]
    k_groups_all: Int[Tensor, " n_iters"] = k_groups_t[valid_mask]
    max_k: int = int(k_groups_all.max().item())

    counts_list: list[Int[Tensor, " max_k"]] = [
        torch.bincount(row[row.ge(0)], minlength=max_k)  # per-iteration cluster sizes
        for row in group_idxs_all
    ]
    counts: Int[Tensor, " n_iters max_k"] = torch.stack(counts_list, dim=0)

    mask_pos: Bool[Tensor, " n_iters max_k"] = counts.gt(0)
    it_idx_t, grp_idx_t = torch.nonzero(mask_pos, as_tuple=True)
    xs_t: Float[Tensor, " n_points"] = it_idx_t.to(torch.float32)
    sizes_t: Float[Tensor, " n_points"] = counts[it_idx_t, grp_idx_t].to(torch.float32)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        xs_t.cpu().numpy(), sizes_t.cpu().numpy(), "bo", markersize=3, alpha=0.15, markeredgewidth=0
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cluster size")
    ax.set_yscale("log")
    ax.set_title("Distribution of cluster sizes over time")

    if file_prefix is not None:
        fig.savefig(f"{file_prefix}_cluster_sizes.{fmt}", bbox_inches="tight", dpi=300)
