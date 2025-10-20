"""Plotting functions for causal importance decision trees."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
) -> None:
    """Plot a list of selected trees by (layer, target_idx, score)."""
    for layer_idx, target_idx, score in picks:
        est = get_estimator_for(models, layer_idx, target_idx)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_prefix}: layer {layer_idx}, target {target_idx}, AP={score:.3f}")
        plot_tree(est, ax=ax, filled=False)  # default styling
        fig.tight_layout()
