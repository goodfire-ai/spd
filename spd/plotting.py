import fnmatch
import math
from collections.abc import Callable
from typing import Literal

import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import (
    Components,
)
from spd.models.sigmoids import SigmoidTypes
from spd.utils.target_ci_solutions import (
    permute_to_dense,
    permute_to_identity,
)


def _plot_causal_importances_figure(
    ci_vals: dict[str, Float[Tensor, "... C"]],
    title_prefix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title_formatter: Callable[[str], str] | None = None,
) -> plt.Figure:
    """Helper function to plot a single mask figure.

    Args:
        ci_vals: Dictionary of causal importances (or causal importances upper leaky relu) to plot
        title_prefix: String to prepend to the title (e.g., "causal importances" or
            "causal importances upper leaky relu")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension
        orientation: The orientation of the subplots
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        The matplotlib figure
    """
    if orientation == "vertical":
        n_rows, n_cols = len(ci_vals), 1
        figsize = (5, 5 * len(ci_vals))
    else:
        n_rows, n_cols = 1, len(ci_vals)
        figsize = (5 * len(ci_vals), 5)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(ci_vals.items()):
        # mask has shape (batch, C) or (batch, pos, C)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        ax = axs[j, 0] if orientation == "vertical" else axs[0, j]
        im = ax.matshow(mask_data, aspect="auto", cmap=colormap)
        images.append(im)

        # Move x-axis ticks to bottom
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Input feature index")

        # Apply custom title formatting if provided
        title = title_formatter(mask_name) if title_formatter is not None else mask_name
        ax.set_title(title)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in ci_vals.values()),
        vmax=max(mask.max().item() for mask in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Capitalize first letter of title prefix for the figure title
    fig.suptitle(f"{title_prefix.capitalize()} - Input magnitude: {input_magnitude}")

    return fig


def get_single_feature_causal_importances(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    sigmoid_type: SigmoidTypes = "leaky_hard",
) -> tuple[dict[str, Float[Tensor, "batch C"]], dict[str, Float[Tensor, "batch C"]]]:
    """Compute causal importance arrays for single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features
        sigmoid_type: Type of sigmoid to use for causal importance calculation

    Returns:
        Tuple of (ci_raw, ci_upper_leaky_raw) dictionaries of causal importance arrays (2D tensors)
    """
    # Create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only use the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=model.target_module_paths
    )[1]

    ci_raw, ci_upper_leaky_raw = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type=sigmoid_type,
        detach_inputs=False,
    )

    return ci_raw, ci_upper_leaky_raw


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    plot_raw_cis: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title_formatter: Callable[[str], str] | None = None,
    sigmoid_type: SigmoidTypes = "leaky_hard",
    identity_patterns: list[str] | None = None,
    dense_patterns: list[str] | None = None,
) -> tuple[dict[str, plt.Figure], dict[str, Float[Tensor, " C"]]]:
    """Plot the values of the causal importances for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        orientation: The orientation of the subplots
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.
        sigmoid_type: Type of sigmoid to use for causal importance calculation.

    Returns:
        Tuple of:
            - Dictionary of figures with keys 'causal_importances' (if plot_raw_cis=True) and 'causal_importances_upper_leaky'
            - Dictionary of permutation indices for causal importances
    """
    # Get the causal importance arrays
    ci_raw, ci_upper_leaky_raw = get_single_feature_causal_importances(
        model=model,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=input_magnitude,
        sigmoid_type=sigmoid_type,
    )

    has_pos_dim = len(batch_shape) == 3

    # Apply permutations based on patterns
    ci = {}
    ci_upper_leaky = {}
    all_perm_indices = {}
    for k in ci_raw:
        # Determine permutation strategy based on patterns
        if identity_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in identity_patterns):
            ci[k], _ = permute_to_identity(ci_vals=ci_raw[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_upper_leaky_raw[k]
            )
        elif dense_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in dense_patterns):
            ci[k], _ = permute_to_dense(ci_vals=ci_raw[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_dense(ci_vals=ci_upper_leaky_raw[k])
        else:
            # Default: identity permutation
            ci[k], _ = permute_to_identity(ci_vals=ci_raw[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_upper_leaky_raw[k]
            )

    # Create figures dictionary
    figures = {}

    if plot_raw_cis:
        ci_fig = _plot_causal_importances_figure(
            ci_vals=ci,
            title_prefix="importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
            orientation=orientation,
            title_formatter=title_formatter,
        )
        figures["causal_importances"] = ci_fig

    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        orientation=orientation,
        title_formatter=title_formatter,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    return figures, all_perm_indices


def plot_subnetwork_attributions_statistics(
    mask: Float[Tensor, "batch_size C"],
) -> dict[str, plt.Figure]:
    """Plot a vertical bar chart of the number of active subnetworks over the batch."""
    batch_size = mask.shape[0]
    if mask.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {mask.ndim}")

    # Sum over subnetworks for each batch entry
    values = mask.sum(dim=1).cpu().detach().numpy()
    bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
    counts, _ = np.histogram(values, bins=bins)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(b) for b in bins[:-1]])
    ax.set_ylabel("Count")
    ax.set_xlabel("Number of active subnetworks")
    ax.set_title("Active subnetworks on current batch")

    # Add value annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_matrix(
    ax: plt.Axes,
    matrix: Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_UV_matrices(
    components: dict[str, Components],
    all_perm_indices: dict[str, Float[Tensor, " C"]] | None = None,
) -> plt.Figure:
    """Plot V and U matrices for each instance, grouped by layer."""
    Vs = {k: v.V for k, v in components.items()}
    Us = {k: v.U for k, v in components.items()}

    n_layers = len(Vs)

    # Create figure for plotting - 2 rows per layer (V and U)
    fig, axs = plt.subplots(
        2 * n_layers,
        1,
        figsize=(5, 5 * 2 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot V and U matrices for each layer
    for j, name in enumerate(sorted(Vs.keys())):
        # Plot V matrix
        V_data = Vs[name]
        if all_perm_indices is not None:
            V_data = V_data[:, all_perm_indices[name]]
        V_data = V_data.detach().cpu().numpy()
        im = axs[2 * j, 0].matshow(V_data, aspect="auto", cmap="coolwarm")
        axs[2 * j, 0].set_ylabel("d_in index")
        axs[2 * j, 0].set_xlabel("Component index")
        axs[2 * j, 0].set_title(f"{name} (V matrix)")
        images.append(im)

        # Plot U matrix
        U_data = Us[name]
        if all_perm_indices is not None:
            U_data = U_data[all_perm_indices[name], :]
        U_data = U_data.detach().cpu().numpy()
        im = axs[2 * j + 1, 0].matshow(U_data, aspect="auto", cmap="coolwarm")
        axs[2 * j + 1, 0].set_ylabel("Component index")
        axs[2 * j + 1, 0].set_xlabel("d_out index")
        axs[2 * j + 1, 0].set_title(f"{name} (U matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = list(Vs.values()) + list(Us.values())
    norm = plt.Normalize(
        vmin=min(M.min().item() for M in all_matrices),
        vmax=max(M.max().item() for M in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return fig


def create_embed_ci_sample_table(
    causal_importances: dict[str, Float[Tensor, "... C"]], key: str, threshold: float
) -> wandb.Table:
    """Create a wandb table visualizing embedding mask values.

    Args:
        causal_importances: Dictionary of causal importances for each component.

    Returns:
        A wandb Table object.
    """
    # Create a 20x10 table for wandb
    table_data = []
    # Add "Row Name" as the first column
    component_names = ["TokenSample"] + ["CompVal" for _ in range(10)]

    for i, ci in enumerate(causal_importances[key][0, :20]):
        active_values = ci[ci > threshold].tolist()
        # Cap at 10 components
        active_values = active_values[:10]
        formatted_values = [f"{val:.2f}" for val in active_values]
        # Pad with empty strings if fewer than 10 components
        while len(formatted_values) < 10:
            formatted_values.append("0")
        # Add row name as the first element
        table_data.append([f"{i}"] + formatted_values)

    return wandb.Table(data=table_data, columns=component_names)


def plot_mean_component_activation_counts(
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]],
) -> plt.Figure:
    """Plots the mean activation counts for each component module in a grid."""
    n_modules = len(mean_component_activation_counts)
    max_cols = 6
    n_cols = min(n_modules, max_cols)
    # Calculate the number of rows needed, rounding up
    n_rows = math.ceil(n_modules / n_cols)

    # Create a figure with the calculated number of rows and columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    # Ensure axs is always a 2D array for consistent indexing, even if n_modules is 1
    axs = axs.flatten()  # Flatten the axes array for easy iteration

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, counts) in enumerate(mean_component_activation_counts.items()):
        ax = axs[i]
        ax.hist(counts.detach().cpu().numpy(), bins=100)
        ax.set_yscale("log")
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Mean Activation Count")
        ax.set_ylabel("Frequency")

    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_modules, n_rows * n_cols):
        axs[i].axis("off")

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()

    return fig


def plot_ci_histograms(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    bins: int = 100,
) -> dict[str, plt.Figure]:
    """Plot histograms of mask values for each layer.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Dictionary mapping layer names to histogram figures.
    """
    fig_dict = {}

    for layer_name_raw, layer_ci in causal_importances.items():
        layer_name = layer_name_raw.replace(".", "_")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(layer_ci.flatten().cpu().numpy(), bins=bins)
        ax.set_title(f"Causal importances for {layer_name}")
        ax.set_xlabel("Causal importance value")
        # Use a log scale
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

        fig_dict[f"mask_vals_{layer_name}"] = fig

    return fig_dict


def plot_component_co_activation_fractions(
    component_co_activation_fractions: dict[str, Float[Tensor, " C C"]],
) -> dict[str, plt.Figure]:
    """Plot the component co-activation fractions for each component module in a grid."""
    n_modules = len(component_co_activation_fractions)

    # Create figure with GridSpec for explicit layout control
    fig = plt.figure(figsize=(8, 8 * n_modules))

    # Create GridSpec: main plots take 85% width, colorbar takes 10%, 5% gap
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)

    # Create a unified normalization for all plots (0 to 1)
    # norm = plt.Normalize(vmin=0, vmax=1)
    norm = plt.Normalize(vmin=0)

    # Store all image objects for colorbar
    images = []

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, fractions) in enumerate(component_co_activation_fractions.items()):
        # Create subplot for the main plot
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(fractions.detach().cpu().numpy(), aspect="auto", cmap="Purples", norm=norm)
        images.append(im)
        ax.set_title(module_name)
        ax.set_xlabel("Denominator Component index")
        ax.set_ylabel("Numerator Component index")

    # Add a unified colorbar for all plots
    if images:
        # Create colorbar in the dedicated right column
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("Co-activation Fraction", fontsize=12)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    return {"component_co_activation_fractions": fig}


def plot_component_abs_left_singular_vectors_cosine_similarity(
    component_abs_left_singular_vectors_cosine_similarity: dict[str, Float[Tensor, " C C"]],
) -> dict[str, plt.Figure]:
    """Plot the cosine similarity between the absolute left singular vectors of the components. (Uses exactsame structure as plotting code for plot_component_co_activation_fractions)"""
    n_modules = len(component_abs_left_singular_vectors_cosine_similarity)
    fig = plt.figure(figsize=(8, 8 * n_modules))
    images = []

    # Create GridSpec: main plots take 85% width, colorbar takes 10%, 5% gap
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)
    for i, (module_name, cosine_similarity) in enumerate(
        component_abs_left_singular_vectors_cosine_similarity.items()
    ):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(cosine_similarity.detach().cpu().numpy(), aspect="auto", cmap="Oranges")
        images.append(im)
        ax.set_title(module_name)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Component index")

    if images:
        # Create colorbar in the dedicated right column
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("Cosine Similarity", fontsize=12)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    return {"component_abs_left_singular_vectors_cosine_similarity": fig}


def plot_cosine_sim_coactivation_correlation(
    cosine_sim_coactivation_correlation: dict[
        str, tuple[Float[Tensor, " C C"], Float[Tensor, " C C"]]
    ],
) -> dict[str, plt.Figure]:
    """Plot the cosine similarity between the absolute left singular vectors of the components. There should be a scatter plot for each module with a histogram along the axes for each variable"""
    n_modules = len(cosine_sim_coactivation_correlation)
    fig = plt.figure(figsize=(8, 8 * n_modules))
    images = []
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)
    for i, (module_name, (cosine_similarity, coactivation_fraction)) in enumerate(
        cosine_sim_coactivation_correlation.items()
    ):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.scatter(
            cosine_similarity.detach().cpu().numpy(),
            coactivation_fraction.detach().cpu().numpy(),
            alpha=0.4,
            s=1.2,
        )
        images.append(im)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.0, 1.05)
        ax.set_title(module_name)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Coactivation Fraction")

    return {"cosine_sim_coactivation_correlation": fig}
