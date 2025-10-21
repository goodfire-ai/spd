"""Plotting functions for causal importance decision trees."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from sklearn.tree import plot_tree

from spd.clustering.ci_dt.core import LayerModel, get_estimator_for


def plot_activations(layers_true: list[np.ndarray], layers_pred: list[np.ndarray]) -> None:
    """Show true and predicted activations as heatmaps."""
    A_true: np.ndarray = np.concatenate(layers_true, axis=1)
    A_pred: np.ndarray = np.concatenate([layers_pred[0]] + layers_pred[1:], axis=1)
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_title("Activations (True)")
    ax1.imshow(A_true, aspect="auto", interpolation="nearest")
    ax1.set_xlabel("components (all layers concatenated)")
    ax1.set_ylabel("samples")
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_title("Activations (Predicted)")
    ax2.imshow(A_pred, aspect="auto", interpolation="nearest")
    ax2.set_xlabel("components (all layers concatenated)")
    ax2.set_ylabel("samples")
    fig1.tight_layout()


def plot_covariance(layers_true: list[np.ndarray]) -> None:
    """Plot covariance between all components across layers."""
    A: np.ndarray = np.concatenate(layers_true, axis=1).astype(float)
    C: np.ndarray = np.cov(A, rowvar=False)
    fig2 = plt.figure(figsize=(6, 6))
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title("Covariance of components (all layers)")
    ax.imshow(C, aspect="auto", interpolation="nearest")
    ax.set_xlabel("component index")
    ax.set_ylabel("component index")
    fig2.tight_layout()


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
