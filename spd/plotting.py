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
from spd.models.components import Components
from spd.models.sigmoids import SigmoidTypes


def permute_to_identity(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Float[Tensor, " C"]]:
    """Permute matrix to make it as close to identity as possible.

    Returns:
        - Permuted mask
        - Permutation indices
    """

    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    effective_rows = min(batch, C)
    perm_indices = torch.zeros(C, dtype=torch.long, device=ci_vals.device)

    perm: list[int] = [0] * C
    used: set[int] = set()
    for i in range(effective_rows):
        sorted_indices: list[int] = torch.argsort(ci_vals[i, :], descending=True).tolist()
        chosen: int = next((col for col in sorted_indices if col not in used), sorted_indices[0])
        perm[i] = chosen
        used.add(chosen)
    remaining: list[int] = sorted(list(set(range(C)) - used))
    for idx, col in enumerate(remaining):
        perm[effective_rows + idx] = col
    new_ci_vals = ci_vals[:, perm]
    perm_indices = torch.tensor(perm, device=ci_vals.device)

    return new_ci_vals, perm_indices


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


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    plot_raw_cis: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title_formatter: Callable[[str], str] | None = None,
    sigmoid_type: SigmoidTypes = "leaky_hard",
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
    # First, create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only plot the mask of the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=model.target_module_paths
    )[1]

    ci_raw, ci_upper_leaky_raw = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        detach_inputs=False,
        sigmoid_type=sigmoid_type,
    )

    ci = {}
    ci_upper_leaky = {}
    all_perm_indices = {}

    for k in ci_raw:
        ci[k], _ = permute_to_identity(ci_vals=ci_raw[k])
        ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(ci_vals=ci_upper_leaky_raw[k])

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
    causal_importances: dict[str, Float[Tensor, "... C"]], key: str
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
        active_values = ci[ci > 0.1].tolist()
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
