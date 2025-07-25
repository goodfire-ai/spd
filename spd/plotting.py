import io
from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import Components
from spd.models.sigmoids import SigmoidTypes


def _render_figure(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


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
    title_formatter: Callable[[str], str] | None = None,
) -> Image.Image:
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
    figsize = (5, 5 * len(ci_vals))
    fig, axs = plt.subplots(
        len(ci_vals),
        1,
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
        ax = axs[j, 0]
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

    img = _render_figure(fig)
    plt.close(fig)

    return img


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    sigmoid_type: SigmoidTypes,
    plot_raw_cis: bool = True,
    title_formatter: Callable[[str], str] | None = None,
) -> tuple[dict[str, Image.Image], dict[str, Float[Tensor, " C"]]]:
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
        sigmoid_type=sigmoid_type,
        detach_inputs=False,
    )

    ci = {}
    ci_upper_leaky = {}
    all_perm_indices = {}

    for k in ci_raw:
        ci[k], _ = permute_to_identity(ci_vals=ci_raw[k])
        ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(ci_vals=ci_upper_leaky_raw[k])

    # Create figures dictionary
    figures: dict[str, Image.Image] = {}

    if plot_raw_cis:
        ci_fig = _plot_causal_importances_figure(
            ci_vals=ci,
            title_prefix="importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
            title_formatter=title_formatter,
        )
        figures["causal_importances"] = ci_fig

    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        title_formatter=title_formatter,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    return figures, all_perm_indices


def plot_subnetwork_attributions_statistics(
    mask: Float[Tensor, "batch_size C"],
) -> dict[str, Image.Image]:
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
    fig_img = _render_figure(fig)

    plt.close(fig)

    return {"subnetwork_attributions_statistics": fig_img}


def plot_UV_matrices(
    components: dict[str, Components],
    all_perm_indices: dict[str, Float[Tensor, " C"]] | None = None,
) -> Image.Image:
    """Plot V and U matrices for each instance, grouped by layer."""
    n_layers = len(components)

    # Create figure for plotting - 2 rows per layer (V and U)
    fig, axs = plt.subplots(
        n_layers,
        2,  # U, V
        figsize=(5 * 2, 5 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot V and U matrices for each layer
    for j, (name, component) in enumerate(sorted(components.items())):
        # Plot V matrix
        V = component.V if all_perm_indices is None else component.V[:, all_perm_indices[name]]
        V_np = V.detach().cpu().numpy()
        im = axs[j, 0].matshow(V_np, aspect="auto", cmap="coolwarm")
        axs[j, 0].set_ylabel("d_in index")
        axs[j, 0].set_xlabel("Component index")
        axs[j, 0].set_title(f"{name} (V matrix)")
        images.append(im)

        # Plot U matrix
        U = component.U if all_perm_indices is None else component.U[all_perm_indices[name], :]
        U_np = U.detach().cpu().numpy()
        im = axs[j, 1].matshow(U_np, aspect="auto", cmap="coolwarm")
        axs[j, 1].set_ylabel("Component index")
        axs[j, 1].set_xlabel("d_out index")
        axs[j, 1].set_title(f"{name} (U matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = [c.V for c in components.values()] + [c.U for c in components.values()]
    norm = plt.Normalize(
        vmin=min(m.min().item() for m in all_matrices),
        vmax=max(m.max().item() for m in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_component_activation_density(
    component_activation_density: dict[str, Float[Tensor, " C"]],
) -> Image.Image:
    n_modules = len(component_activation_density)

    fig, axs = plt.subplots(
        n_modules,
        1,
        figsize=(5, 5 * n_modules),
        squeeze=False,
    )

    axs = axs.flatten()  # Flatten the axes array for easy iteration

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, density) in enumerate(component_activation_density.items()):
        ax = axs[i]
        ax.hist(density.detach().cpu().numpy(), bins=100)
        ax.set_yscale("log")
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_ci_histograms(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    bins: int = 100,
) -> Image.Image:
    """Plot histograms of mask values for all layers in a single figure.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Single figure with subplots for each layer.
    """
    n_layers = len(causal_importances)

    fig, axs = plt.subplots(
        n_layers,
        1,
        figsize=(6, 5 * n_layers),
        squeeze=False,
    )

    axs = axs.flatten()  # Flatten the axes array for easy iteration

    for i, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
        layer_name = layer_name_raw.replace(".", "_")
        ax = axs[i]
        ax.hist(layer_ci.flatten().cpu().numpy(), bins=bins)
        ax.set_title(f"Causal importances for {layer_name}")
        ax.set_xlabel("Causal importance value")
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

    # Hide unused subplots
    for i in range(n_layers, n_layers):
        axs[i].axis("off")

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img
