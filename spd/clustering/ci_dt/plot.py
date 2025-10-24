"""Plotting functions for causal importance decision trees."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from sklearn.tree import plot_tree
import torch

from spd.clustering.ci_dt.core import LayerModel, MetricKey, get_estimator_for

METRIC_DISPLAY_INFO: dict[MetricKey, dict[str, str]] = {
    "ap": {
        "ylabel": "Average Precision",
        "title": (
            r"Average Precision per Target Component" + "\n"
            r"$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$ where "
            r"$P_n = \frac{\text{TP}}{\text{TP}+\text{FP}}$, "
            r"$R_n = \frac{\text{TP}}{\text{TP}+\text{FN}}$"
        ),
        "color": "C0",
    },
    "acc": {
        "ylabel": "Accuracy",
        "title": (
            r"Accuracy per Target Component" + "\n"
            r"$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$"
        ),
        "color": "C1",
    },
    "bacc": {
        "ylabel": "Balanced Accuracy",
        "title": (
            r"Balanced Accuracy per Target Component" + "\n"
            r"$\text{Balanced Acc} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP}+\text{FN}} + \frac{\text{TN}}{\text{TN}+\text{FP}}\right)$"
        ),
        "color": "C2",
    },
    "prev": {
        "ylabel": "Prevalence",
        "title": (
            r"Component Prevalence" + "\n"
            r"$\text{Prevalence} = \frac{\text{TP}+\text{FN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}$ (fraction of samples where component is active)"
        ),
        "color": "C5",
    },
    "tpr": {
        "ylabel": "TPR",
        "title": (
            r"True Positive Rate (TPR / Recall / Sensitivity)" + "\n"
            r"$\text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}}$ (how well we predict active components)"
        ),
        "color": "C0",
    },
    "tnr": {
        "ylabel": "TNR",
        "title": (
            r"True Negative Rate (TNR / Specificity)" + "\n"
            r"$\text{TNR} = \frac{\text{TN}}{\text{TN}+\text{FP}}$ (how well we predict inactive components)"
        ),
        "color": "C1",
    },
    "precision": {
        "ylabel": "Precision",
        "title": (
            r"Precision (Positive Predictive Value)" + "\n"
            r"$\text{PPV} = \frac{\text{TP}}{\text{TP}+\text{FP}}$ (when we predict active, how often are we right?)"
        ),
        "color": "C2",
    },
    "npv": {
        "ylabel": "NPV",
        "title": (
            r"Negative Predictive Value (NPV)" + "\n"
            r"$\text{NPV} = \frac{\text{TN}}{\text{TN}+\text{FN}}$ (when we predict inactive, how often are we right?)"
        ),
        "color": "C3",
    },
    "f1": {
        "ylabel": "F1 Score",
        "title": (
            r"F1 Score per Target Component" + "\n"
            r"$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (harmonic mean)"
        ),
        "color": "C4",
    },
}


def _plot_metric_scatter(
    ax: plt.Axes,
    per_layer_stats: list[dict[str, Any]],
    module_keys: list[str],
    metric_key: MetricKey,
    jitter_amount: float = 0.15,
) -> None:
    """Helper function to plot jittered scatter with mean lines for a metric.

    Handles all formatting including axis labels, ticks, grid, and module name cleaning.
    Display properties (title, ylabel, color) are looked up from METRIC_DISPLAY_INFO.

    Args:
        ax: Matplotlib axis to plot on
        per_layer_stats: List of dicts with metrics per layer
        module_keys: List of module names for x-axis labels
        metric_key: Key for metric (e.g., "tpr", "npv", "f1")
        jitter_amount: Amount of horizontal jitter for scatter points
    """
    # Look up display properties
    display_info = METRIC_DISPLAY_INFO[metric_key]
    mean_key = f"mean_{metric_key}"
    ylabel = display_info["ylabel"]
    title = display_info["title"]
    color = display_info["color"]

    L: int = len(per_layer_stats)
    np.random.seed(42)

    # Plot scatter and means
    for layer_idx, stats in enumerate(per_layer_stats):
        values: np.ndarray = stats[metric_key]
        valid: np.ndarray = values[~np.isnan(values)]
        if len(valid) > 0:
            x_positions: np.ndarray = np.ones(len(valid)) * (layer_idx + 1)
            x_jittered: np.ndarray = x_positions + np.random.uniform(
                -jitter_amount, jitter_amount, len(valid)
            )
            ax.scatter(x_jittered, valid, alpha=0.5, s=20, color=color, edgecolors="none")
            ax.plot(
                [layer_idx + 1 - 0.3, layer_idx + 1 + 0.3],
                [stats[mean_key], stats[mean_key]],
                "r-",
                linewidth=2,
                label="Mean" if layer_idx == 0 else "",
            )

    # Clean module names
    clean_keys = [k.removeprefix("model.layers.").replace("_proj", "") for k in module_keys]

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("Target Module")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, L + 1))
    ax.set_xticklabels(clean_keys[1 : L + 1], rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()


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
    avg_sim: Float[np.ndarray, n] = similarity.mean(axis=1)
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


def add_component_labeling(ax: plt.Axes, component_labels: list[str], axis: str = "x") -> None:
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
    A: Float[np.ndarray, "n_samples n_components"] = np.concatenate(layers_true, axis=1).astype(
        float
    )

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


def plot_metric(
    per_layer_stats: list[dict[str, Any]],
    module_keys: list[str],
    metric_key: MetricKey,
) -> None:
    """Plot distribution of a metric per layer with scatter plot and jitter.

    Display properties (title, ylabel, color) are looked up from METRIC_DISPLAY_INFO.

    Args:
        per_layer_stats: List of dicts with metrics per layer
        module_keys: List of module names for x-axis labels
        metric_key: Key for metric to plot (e.g., "tpr", "npv", "f1", "ap", "bacc")
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_metric_scatter(
        ax=ax,
        per_layer_stats=per_layer_stats,
        module_keys=module_keys,
        metric_key=metric_key,
    )
    fig.tight_layout()


def plot_ap_vs_prevalence(per_layer_stats: list[dict[str, Any]], models: list[LayerModel]) -> None:
    """Plot AP vs prevalence scatter colored by tree depth.

    Args:
        per_layer_stats: List of dicts with metrics per layer
        models: List of trained LayerModel objects (needed for tree depths)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    prevalence_list: list[float] = []
    ap_list: list[float] = []
    depth_list: list[int] = []

    for _layer_idx, (stats, model) in enumerate(zip(per_layer_stats, models, strict=True)):
        for target_idx, (prev, ap) in enumerate(zip(stats["prev"], stats["ap"], strict=True)):
            if not np.isnan(ap):
                prevalence_list.append(prev)
                ap_list.append(ap)
                # Get tree depth for this target
                estimator = model.model.estimators_[target_idx]
                depth_list.append(int(estimator.tree_.max_depth))

    prevalence_arr: np.ndarray = np.array(prevalence_list)
    ap_arr: np.ndarray = np.array(ap_list)
    depth_arr: np.ndarray = np.array(depth_list)

    # Plot baseline: for uncorrelated variables, expected AP = prevalence
    prev_range: np.ndarray = np.logspace(np.log10(prevalence_arr.min()), np.log10(prevalence_arr.max()), 100)
    ax.plot(prev_range, prev_range, 'k--', alpha=0.5, linewidth=1.5, label='Random baseline (AP = prevalence)', zorder=1)

    scatter = ax.scatter(
        prevalence_arr,
        ap_arr,
        c=depth_arr,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="none",
        linewidths=0,
        zorder=2,
    )

    ax.set_title(
        r"Average Precision vs Component Prevalence" + "\n"
        r"$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$ where $P_n = \frac{\text{TP}}{\text{TP}+\text{FP}}$, $R_n = \frac{\text{TP}}{\text{TP}+\text{FN}}$" + "\n"
        r"Colored by tree depth"
    )
    ax.set_xlabel("Prevalence (log scale)")
    ax.set_ylabel("Average Precision")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Tree Depth")

    fig.tight_layout()


def plot_component_activity_breakdown(
    component_acts: dict[str, np.ndarray|torch.Tensor],
    module_keys: list[str],
    activation_threshold: float,
    logy: bool = False,
) -> None:
    """Plot stacked bar chart of component activity breakdown per module.

    Args:
        component_acts: Dict of continuous activations per module
        module_keys: List of module names for x-axis labels
        activation_threshold: Threshold used for binary conversion
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute counts for each module
    n_varying_list: list[int] = []
    n_always_dead_list: list[int] = []
    n_always_alive_list: list[int] = []

    for module_key in module_keys:
        acts: np.ndarray = component_acts[module_key]
        # Convert to numpy if needed
        if hasattr(acts, "cpu"):
            acts = acts.cpu().numpy()

        # Flatten if 3D (batch, seq_len, n_components) -> (batch*seq_len, n_components)
        # This treats each token position as a separate sample, consistent with decision tree training
        if acts.ndim == 3:
            acts = acts.reshape(-1, acts.shape[-1])

        # Convert to boolean
        acts_bool: np.ndarray = (acts >= activation_threshold).astype(bool)

        # Count each category
        always_dead: np.ndarray = ~acts_bool.any(axis=0)
        always_alive: np.ndarray = acts_bool.all(axis=0)
        varying: np.ndarray = ~(always_dead | always_alive)

        n_always_dead_list.append(int(always_dead.sum()))
        n_always_alive_list.append(int(always_alive.sum()))
        n_varying_list.append(int(varying.sum()))

    # Convert to arrays
    n_varying: np.ndarray = np.array(n_varying_list)
    n_always_dead: np.ndarray = np.array(n_always_dead_list)
    n_always_alive: np.ndarray = np.array(n_always_alive_list)

    # For each module, sort the three categories by size (smallest to largest)
    # This will be stacked bottom-to-top as smallest, medium, largest
    x_pos: np.ndarray = np.arange(len(module_keys))

    # For each position, we need to stack in order of size
    # We'll plot all bars for the smallest category first, then medium, then largest
    for module_idx in range(len(module_keys)):
        # Get values for this module
        vals: list[tuple[float, str, str]] = [
            (n_varying[module_idx], "Varying", "C2"),
            (n_always_alive[module_idx], "Always Active", "C1"),
            (n_always_dead[module_idx], "Always Inactive", "C0"),
        ]
        # Sort by value (smallest to largest)
        vals.sort(key=lambda x: x[0])

        # Stack them
        bottom: float = 0
        for val, label, color in vals:
            if val > 0:  # Only plot if non-zero
                ax.bar(
                    module_idx,
                    val,
                    bottom=bottom,
                    color=color,
                    label=label if module_idx == 0 else "",  # Only label once
                )
                bottom += val

    ax.set_title("Component Activity Distribution per Module")
    ax.set_xlabel("Module")
    ax.set_ylabel("Number of Components (log scale)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_keys, rotation=45, ha="right")
    if logy:
        ax.set_yscale("log")

    # Create legend with correct labels
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="C2", label="Varying"),
        Patch(facecolor="C1", label="Always Active"),
        Patch(facecolor="C0", label="Always Inactive"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()


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
) -> dict[str, Float[np.ndarray, " n_trees"]]:
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


def plot_tree_statistics(models: list[LayerModel], per_layer_stats: list[dict[str, Any]]) -> None:
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
    depth_bins: Int[np.ndarray, " n_bins"] = np.arange(
        int(stats["depth"].min()), int(stats["depth"].max()) + 2
    )
    acc_bins: Float[np.ndarray, " n_bins"] = np.linspace(0, 1, 11)
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
    leaf_bins: Int[np.ndarray, " n_bins"] = np.linspace(
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

    # Heatmap: AP vs prevalence
    # Need to compute prevalence for each tree from per_layer_stats
    prevalence_list: list[float] = []
    ap_list_for_heatmap: list[float] = []

    for layer_stats in per_layer_stats:
        for prev, ap in zip(layer_stats["prev"], layer_stats["ap"], strict=True):
            if not np.isnan(ap):
                prevalence_list.append(prev)
                ap_list_for_heatmap.append(ap)

    prevalence_arr: np.ndarray = np.array(prevalence_list)
    ap_arr_for_heatmap: np.ndarray = np.array(ap_list_for_heatmap)

    # Prevalence bins (log scale)
    prev_min: float = max(prevalence_arr.min(), 1e-4)  # Avoid log(0)
    prev_max: float = prevalence_arr.max()
    prev_bins: Float[np.ndarray, " n_bins"] = np.logspace(
        np.log10(prev_min), np.log10(prev_max), 10
    )

    # AP bins (linear)
    ap_bins_heatmap: Float[np.ndarray, " n_bins"] = np.linspace(0, 1, 11)

    heatmap_prev_ap: Float[np.ndarray, "prev_bins ap_bins"]
    heatmap_prev_ap, _, _ = np.histogram2d(
        prevalence_arr, ap_arr_for_heatmap, bins=[prev_bins, ap_bins_heatmap]
    )

    fig7, ax7 = plt.subplots(figsize=(8, 6))
    heatmap_log = np.log10(heatmap_prev_ap.T + 1)
    im = ax7.imshow(heatmap_log, origin="lower", aspect="auto", cmap="Blues")

    # X-axis: prevalence (log scale)
    ax7.set_xticks(range(len(prev_bins) - 1))
    ax7.set_xticklabels([f"{x:.3f}" for x in prev_bins[:-1]], rotation=45, ha="right")
    ax7.set_xlabel("Prevalence (log scale)")

    # Y-axis: AP
    ax7.set_yticks(range(len(ap_bins_heatmap) - 1))
    ax7.set_yticklabels([f"{x:.1f}" for x in ap_bins_heatmap[:-1]])
    ax7.set_ylabel("Average Precision")

    ax7.set_title(
        r"Tree Performance vs Component Prevalence" + "\n"
        r"AP = Average Precision, Prev = $\frac{n_{\text{active}}}{n_{\text{total}}}$"
    )

    # Add counts to cells
    for i in range(len(prev_bins) - 1):
        for j in range(len(ap_bins_heatmap) - 1):
            count = int(heatmap_prev_ap[i, j])
            if count > 0:
                ax7.text(i, j, str(count), ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax7, label="log10(count+1)")
    fig7.tight_layout()
