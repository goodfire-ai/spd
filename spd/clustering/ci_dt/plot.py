"""Plotting functions for causal importance decision trees."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float, Int
from sklearn.tree import plot_tree

from spd.clustering.ci_dt.core import LayerModel, get_estimator_for


def greedy_sort(A: np.ndarray, axis: int) -> np.ndarray:
    """Greedy ordering by cosine similarity.

    Starts from the most central item (highest average similarity to all others)
    and greedily adds the nearest neighbor at each step.

    Args:
        A: 2D array to sort
        axis: 0 to sort rows (samples), 1 to sort columns (components)

    Returns:
        Array of indices in sorted order
    """
    # Transpose if sorting columns
    if axis == 1:
        A = A.T

    # Compute cosine similarity
    norms: Float[np.ndarray, "n 1"] = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)  # Avoid division by zero
    A_normalized: Float[np.ndarray, "n d"] = A / norms
    similarity: Float[np.ndarray, "n n"] = A_normalized @ A_normalized.T

    # Start from most central item (highest average similarity)
    n: int = similarity.shape[0]
    avg_sim: Float[np.ndarray, "n"] = similarity.mean(axis=1)
    start_idx: int = int(np.argmax(avg_sim))

    # Greedy ordering: always add nearest unvisited neighbor
    ordered: list[int] = [start_idx]
    remaining: set[int] = set(range(n))
    remaining.remove(start_idx)
    current: int = start_idx

    while remaining:
        # Find unvisited item with highest similarity to current
        best_sim: float = -1.0
        best_idx: int = -1
        for idx in remaining:
            sim: float = float(similarity[current, idx])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        ordered.append(best_idx)
        remaining.remove(best_idx)
        current = best_idx

    return np.array(ordered, dtype=np.int64)


def add_component_labeling(
    ax: plt.Axes, component_labels: list[str], axis: str = "x"
) -> None:
    """Add component labeling using major/minor ticks to show module boundaries.

    Args:
        ax: Matplotlib axis to modify
        component_labels: List of component labels in format "module:index"
        axis: Which axis to label ('x' or 'y')
    """
    if not component_labels:
        return

    # Extract module information
    module_changes: list[int] = []
    current_module: str = component_labels[0].split(":")[0]
    module_labels: list[str] = []

    for i, label in enumerate(component_labels):
        module: str = label.split(":")[0]
        if module != current_module:
            module_changes.append(i)
            module_labels.append(current_module)
            current_module = module
    module_labels.append(current_module)

    # Set up major and minor ticks
    # Minor ticks: every 10 components
    minor_ticks: list[int] = list(range(0, len(component_labels), 10))

    # Major ticks: module boundaries (start of each module)
    major_ticks: list[int] = [0] + module_changes
    major_labels_final: list[str] = module_labels

    if axis == "x":
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels_final, rotation=45, ha="right")
        ax.set_xlim(-0.5, len(component_labels) - 0.5)
        # Style the ticks
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
        ax.tick_params(axis="x", which="major", length=6, width=1.5)
        for x in major_ticks:
            ax.axvline(x - 0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticklabels(major_labels_final)
        ax.set_ylim(-0.5, len(component_labels) - 0.5)
        # Style the ticks
        ax.tick_params(axis="y", which="minor", length=2, width=0.5)
        ax.tick_params(axis="y", which="major", length=6, width=1.5)
        for y in major_ticks:
            ax.axhline(y - 0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_activations(
    layers_true: list[np.ndarray],
    layers_pred: list[np.ndarray],
    module_keys: list[str],
    activation_threshold: float,
    sample_order: np.ndarray | None = None,
) -> None:
    """Plot true and predicted activations with optional sorting and diff.

    Args:
        layers_true: List of boolean activation arrays per layer
        layers_pred: List of predicted activation arrays per layer
        module_keys: List of module names (e.g., ["blocks.0.attn.W_Q", ...])
        activation_threshold: Threshold used for binary conversion
        sample_order: Optional array of sample indices for sorting. If None, plots unsorted.
    """
    A_true: Float[np.ndarray, "n_samples n_components"] = np.concatenate(
        layers_true, axis=1
    ).astype(float)
    A_pred: Float[np.ndarray, "n_samples n_components"] = np.concatenate(
        layers_pred, axis=1
    ).astype(float)

    # Apply sample ordering if provided
    if sample_order is not None:
        A_true = A_true[sample_order, :]
        A_pred = A_pred[sample_order, :]
        sorted_label: str = " (Sorted by Sample Similarity)"
        xlabel: str = "Sample index (sorted)"
    else:
        sorted_label = ""
        xlabel = "Sample index"

    # Create component labels for unsorted plots
    component_labels: list[str] | None = None
    if sample_order is None:
        component_labels = []
        for module_key, layer in zip(module_keys, layers_true, strict=True):
            n_components: int = layer.shape[1]
            component_labels.extend([f"{module_key}:{i}" for i in range(n_components)])

    # Determine number of subplots
    n_plots: int = 3 if sample_order is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    # Plot true activations
    ax1.imshow(A_true.T, aspect="auto", interpolation="nearest", cmap="Blues")
    ax1.set_title(
        rf"True Binary Activations{sorted_label}" + "\n"
        r"$A_{ij} = \mathbb{1}[\text{activation}_{ij} > \theta]$, "
        rf"$\theta = {activation_threshold}$"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Component index")
    if component_labels is not None:
        add_component_labeling(ax1, component_labels, axis="y")

    # Plot predicted activations
    ax2.imshow(A_pred.T, aspect="auto", interpolation="nearest", cmap="Reds")
    ax2.set_title(
        rf"Predicted Binary Activations{sorted_label}" + "\n"
        r"$\hat{A}_{ij} = \mathbb{1}[P(A_{ij}=1) > 0.5]$"
    )
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Component index")
    if component_labels is not None:
        add_component_labeling(ax2, component_labels, axis="y")

    # Add diff plot if sorted
    if sample_order is not None:
        A_diff: Float[np.ndarray, "n_samples n_components"] = A_pred - A_true
        im3 = ax3.imshow(
            A_diff.T, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1
        )
        ax3.set_title(
            r"Prediction Errors (Predicted - True)" + "\n"
            r"Red = FP ($\hat{A}=1, A=0$), Blue = FN ($\hat{A}=0, A=1$), White = Correct"
        )
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Component index")
        plt.colorbar(im3, ax=ax3, label="Error")

    fig.tight_layout()


def plot_covariance(
    layers_true: list[np.ndarray],
    module_keys: list[str],
    component_order: np.ndarray | None = None,
) -> None:
    """Plot covariance matrix with optional component ordering.

    Args:
        layers_true: List of boolean activation arrays per layer
        module_keys: List of module names for labeling
        component_order: Optional array of component indices for sorting. If None, plots unsorted.
    """
    A: Float[np.ndarray, "n_samples n_components"] = np.concatenate(
        layers_true, axis=1
    ).astype(float)

    # Apply component ordering if provided
    if component_order is not None:
        A = A[:, component_order]
        sorted_label: str = " (Sorted by Component Similarity)"
        xlabel: str = "Component index (sorted)"
        ylabel: str = "Component index (sorted)"
    else:
        sorted_label = ""
        xlabel = "Component index"
        ylabel = "Component index"

    # Compute covariance
    C: Float[np.ndarray, "n_components n_components"] = np.cov(A, rowvar=False)

    # Center colormap on 0
    vmax: float = float(np.abs(C).max())
    vmin: float = -vmax

    # Create component labels for unsorted plots
    component_labels: list[str] | None = None
    if component_order is None:
        component_labels = []
        for module_key, layer in zip(module_keys, layers_true, strict=True):
            n_components: int = layer.shape[1]
            component_labels.extend([f"{module_key}:{i}" for i in range(n_components)])

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(C, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(
        rf"Component Covariance Matrix{sorted_label}" + "\n"
        r"$\text{Cov}(i,j) = \mathbb{E}[(A_i - \mu_i)(A_j - \mu_j)]$" + "\n"
        r"where $A_i$ is binary activation of component $i$"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add layer boundaries for unsorted
    if component_labels is not None:
        add_component_labeling(ax, component_labels, axis="x")
        add_component_labeling(ax, component_labels, axis="y")

    plt.colorbar(im, ax=ax, label="Covariance")
    fig.tight_layout()


def plot_layer_metrics(per_layer_stats: list[dict[str, Any]]) -> None:
    """Plot summary metrics per layer and per-target AP vs prevalence."""
    L: int = len(per_layer_stats)
    mean_ap: np.ndarray = np.array([d["mean_ap"] for d in per_layer_stats])
    mean_acc: np.ndarray = np.array([d["mean_acc"] for d in per_layer_stats])
    mean_bacc: np.ndarray = np.array([d["mean_bacc"] for d in per_layer_stats])

    # bar: mean AP, ACC, BACC per layer (three separate figures to respect one-plot rule)
    fig3 = plt.figure(figsize=(8, 3))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_title("Mean Average Precision per layer")
    ax3.bar(np.arange(1, L + 1), mean_ap)
    ax3.set_xlabel("layer index (target)")
    ax3.set_ylabel("mean AP")
    fig3.tight_layout()

    fig4 = plt.figure(figsize=(8, 3))
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.set_title("Mean Accuracy per layer")
    ax4.bar(np.arange(1, L + 1), mean_acc)
    ax4.set_xlabel("layer index (target)")
    ax4.set_ylabel("mean accuracy")
    fig4.tight_layout()

    fig5 = plt.figure(figsize=(8, 3))
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.set_title("Mean Balanced Accuracy per layer")
    ax5.bar(np.arange(1, L + 1), mean_bacc)
    ax5.set_xlabel("layer index (target)")
    ax5.set_ylabel("mean balanced accuracy")
    fig5.tight_layout()

    # scatter: prevalence vs AP for all targets across layers
    fig6 = plt.figure(figsize=(6, 5))
    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.set_title("Per-target AP vs prevalence")
    x_list: list[float] = []
    y_list: list[float] = []
    for d in per_layer_stats:
        x_list.extend(list(d["prev"]))
        y_list.extend(list(d["ap"]))
    ax6.scatter(x_list, y_list, alpha=0.6)
    ax6.set_xlabel("prevalence")
    ax6.set_ylabel("average precision")
    fig6.tight_layout()


def plot_selected_trees(
    picks: list[tuple[int, int, float]],
    title_prefix: str,
    models: list[LayerModel],
    feature_names: list[list[str]] | None = None,
) -> None:
    """Plot a list of selected trees by (layer, target_idx, score).

    Args:
        picks: List of (layer_idx, target_idx, score) tuples identifying trees to plot
        title_prefix: Prefix for plot titles (e.g. "Best" or "Worst")
        models: Trained LayerModel objects
        feature_names: Optional list of feature name lists, one per layer.
                      feature_names[k] contains names for all features used to predict layer k.
    """
    for layer_idx, target_idx, score in picks:
        est = get_estimator_for(models, layer_idx, target_idx)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_prefix}: layer {layer_idx}, target {target_idx}, AP={score:.3f}")

        # Get feature names for this layer if available
        feat_names = None
        if feature_names is not None and 0 <= layer_idx < len(feature_names):
            feat_names = feature_names[layer_idx]

        plot_tree(est, ax=ax, filled=False, feature_names=feat_names)
        fig.tight_layout()


def extract_tree_stats(
    models: list[LayerModel],
    per_layer_stats: list[dict[str, Any]],
) -> dict[str, Float[np.ndarray, "n_trees"]]:
    """Extract depth, leaf count, and accuracy for all trees across all layers."""
    depths: list[int] = []
    leaf_counts: list[int] = []
    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    aps: list[float] = []

    for lm, stats in zip(models, per_layer_stats, strict=True):
        for i, estimator in enumerate(lm.model.estimators_):
            depths.append(int(estimator.tree_.max_depth))
            leaf_counts.append(int(estimator.tree_.n_leaves))
            accuracies.append(float(stats["acc"][i]))
            balanced_accuracies.append(float(stats["bacc"][i]))
            aps.append(float(stats["ap"][i]))

    return {
        "depth": np.array(depths),
        "n_leaves": np.array(leaf_counts),
        "accuracy": np.array(accuracies),
        "balanced_accuracy": np.array(balanced_accuracies),
        "ap": np.array(aps),
    }


def plot_tree_statistics(
    models: list[LayerModel], per_layer_stats: list[dict[str, Any]]
) -> None:
    """Plot distributions of tree depth, leaf count, and their correlations with accuracy."""
    stats = extract_tree_stats(models, per_layer_stats)

    # Distribution of tree depths
    fig1, ax1 = plt.subplots()
    ax1.hist(stats["depth"], bins=range(int(stats["depth"].max()) + 2))
    ax1.set_yscale("log")
    ax1.set_xlabel("Tree depth")
    ax1.set_ylabel("Count (log scale)")

    # Distribution of leaf counts
    fig2, ax2 = plt.subplots()
    ax2.hist(stats["n_leaves"], bins=50)
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of leaves")
    ax2.set_ylabel("Count (log scale)")

    # Distribution of accuracies
    fig3, ax3 = plt.subplots()
    ax3.hist(stats["accuracy"][~np.isnan(stats["accuracy"])], bins=30)
    ax3.set_yscale("log")
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Count (log scale)")

    # Heatmap: depth vs accuracy
    valid_mask: np.ndarray = ~np.isnan(stats["accuracy"])
    depth_bins: Int[np.ndarray, "n_bins"] = np.arange(
        int(stats["depth"].min()), int(stats["depth"].max()) + 2
    )
    acc_bins: Float[np.ndarray, "n_bins"] = np.linspace(0, 1, 11)
    heatmap_depth_acc: Float[np.ndarray, "depth_bins acc_bins"]
    heatmap_depth_acc, _, _ = np.histogram2d(
        stats["depth"][valid_mask], stats["accuracy"][valid_mask], bins=[depth_bins, acc_bins]
    )

    fig4, ax4 = plt.subplots()
    heatmap_log: Float[np.ndarray, "depth_bins acc_bins"] = np.log10(
        heatmap_depth_acc.T + 1
    )  # +1 to avoid log(0)
    im = ax4.imshow(heatmap_log, origin="lower", aspect="auto", cmap="Blues")
    ax4.set_xticks(range(len(depth_bins) - 1))
    ax4.set_xticklabels(depth_bins[:-1])
    ax4.set_yticks(range(len(acc_bins) - 1))
    ax4.set_yticklabels([f"{x:.1f}" for x in acc_bins[:-1]])
    ax4.set_xlabel("Tree depth")
    ax4.set_ylabel("Accuracy")
    for i in range(len(depth_bins) - 1):
        for j in range(len(acc_bins) - 1):
            count: int = int(heatmap_depth_acc[i, j])
            if count > 0:
                ax4.text(i, j, str(count), ha="center", va="center")
    plt.colorbar(im, ax=ax4, label="log10(count+1)")

    # Heatmap: leaf count vs accuracy
    leaf_bins: Int[np.ndarray, "n_bins"] = np.linspace(
        int(stats["n_leaves"].min()), int(stats["n_leaves"].max()) + 1, 11, dtype=int
    )
    heatmap_leaf_acc: Float[np.ndarray, "leaf_bins acc_bins"]
    heatmap_leaf_acc, _, _ = np.histogram2d(
        stats["n_leaves"][valid_mask], stats["accuracy"][valid_mask], bins=[leaf_bins, acc_bins]
    )

    fig5, ax5 = plt.subplots()
    heatmap_log = np.log10(heatmap_leaf_acc.T + 1)
    im = ax5.imshow(heatmap_log, origin="lower", aspect="auto", cmap="Blues")
    ax5.set_xticks(range(len(leaf_bins) - 1))
    ax5.set_xticklabels(leaf_bins[:-1])
    ax5.set_yticks(range(len(acc_bins) - 1))
    ax5.set_yticklabels([f"{x:.1f}" for x in acc_bins[:-1]])
    ax5.set_xlabel("Number of leaves")
    ax5.set_ylabel("Accuracy")
    for i in range(len(leaf_bins) - 1):
        for j in range(len(acc_bins) - 1):
            count: int = int(heatmap_leaf_acc[i, j])
            if count > 0:
                ax5.text(i, j, str(count), ha="center", va="center")
    plt.colorbar(im, ax=ax5, label="log10(count+1)")

    # Heatmap: depth vs leaf count
    heatmap_depth_leaf: Float[np.ndarray, "depth_bins leaf_bins"]
    heatmap_depth_leaf, _, _ = np.histogram2d(
        stats["depth"][valid_mask], stats["n_leaves"][valid_mask], bins=[depth_bins, leaf_bins]
    )

    fig6, ax6 = plt.subplots()
    heatmap_log = np.log10(heatmap_depth_leaf.T + 1)
    im = ax6.imshow(heatmap_log, origin="lower", aspect="auto", cmap="Blues")
    ax6.set_xticks(range(len(depth_bins) - 1))
    ax6.set_xticklabels(depth_bins[:-1])
    ax6.set_yticks(range(len(leaf_bins) - 1))
    ax6.set_yticklabels(leaf_bins[:-1])
    ax6.set_xlabel("Tree depth")
    ax6.set_ylabel("Number of leaves")
    for i in range(len(depth_bins) - 1):
        for j in range(len(leaf_bins) - 1):
            count: int = int(heatmap_depth_leaf[i, j])
            if count > 0:
                ax6.text(i, j, str(count), ha="center", va="center")
    plt.colorbar(im, ax=ax6, label="log10(count+1)")
