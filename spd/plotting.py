import fnmatch
import io
from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import (
    Components,
)
from spd.models.sigmoids import SigmoidTypes
from spd.utils.target_ci_solutions import permute_to_dense, permute_to_identity


def _render_figure(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


def _plot_causal_importances_figure(
    ci_vals: dict[str, Float[Tensor, "... C"]],
    title_prefix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
    title_formatter: Callable[[str], str] | None = None,
) -> Image.Image:
    """Plot causal importances for components stacked vertically.

    Args:
        ci_vals: Dictionary of causal importances (or causal importances upper leaky relu) to plot
        title_prefix: String to prepend to the title (e.g., "causal importances" or
            "causal importances upper leaky relu")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension
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

    pre_weight_acts = model(
        batch, mode="pre_forward_cache", module_names=model.target_module_paths
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
    sigmoid_type: SigmoidTypes,
    identity_patterns: list[str] | None = None,
    dense_patterns: list[str] | None = None,
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

    ci: dict[str, Float[Tensor, "... C"]] = {}
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]] = {}
    all_perm_indices: dict[str, Float[Tensor, " C"]] = {}
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
    figures: dict[str, Image.Image] = {}

    # TODO: Need to handle this differently for e.g. convolutional tasks
    has_pos_dim = len(batch_shape) == 3
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
    bins: int = 100,
) -> Image.Image:
    """Plot the activation density of each component as a histogram, stacked vertically."""

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
        data = density.detach().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_ci_values_histograms(
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

        data = layer_ci.flatten().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(f"Causal importances for {layer_name}")
        ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_component_co_activation_fractions(
    component_co_activation_fractions: dict[str, Float[Tensor, " C C"]],
) -> Image.Image:
    """Plot the component co-activation fractions for each component module in a grid."""
    n_modules = len(component_co_activation_fractions)

    # Create figure with GridSpec for explicit layout control
    fig = plt.figure(figsize=(8, 8 * n_modules))

    # Create GridSpec: main plots take 85% width, colorbar takes 10%, 5% gap
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)
    norm = plt.Normalize(vmin=0, vmax=1)

    # Store all image objects for colorbar
    images = []

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, fractions) in enumerate(component_co_activation_fractions.items()):
        # Create subplot for the main plot
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(fractions.detach().cpu().numpy(), aspect="auto", cmap="Blues", norm=norm)
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

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_component_abs_left_singular_vectors_geometric_interaction_strengths(
    component_abs_left_singular_vectors_geometric_interaction_strengths: dict[
        str, Float[Tensor, " C C"]
    ],
) -> Image.Image:
    """Plot the cosine similarity between the absolute left singular vectors of the components. (Uses exactsame structure as plotting code for plot_component_co_activation_fractions)"""
    n_modules = len(component_abs_left_singular_vectors_geometric_interaction_strengths)
    fig = plt.figure(figsize=(8, 8 * n_modules))
    images = []

    # Create GridSpec: main plots take 85% width, colorbar takes 10%, 5% gap
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)

    norm = plt.Normalize(vmin=0, vmax=1)

    for i, (module_name, geometric_interaction_strength) in enumerate(
        component_abs_left_singular_vectors_geometric_interaction_strengths.items()
    ):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(
            geometric_interaction_strength.detach().cpu().numpy(),
            aspect="auto",
            cmap="Reds",
            norm=norm,
        )
        images.append(im)
        ax.set_title(module_name)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Component index")

    if images:
        # Create colorbar in the dedicated right column, ensuring the min colour and max colour are -1 and 1
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("Geometric Interaction Strength", fontsize=12)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_geometric_interaction_strength_vs_coactivation(
    geometric_interaction_strength_vs_coactivation_data: dict[
        str, tuple[Float[Tensor, " C C"], Float[Tensor, " C C"]]
    ],
) -> Image.Image:
    """Plot the cosine similarity between the absolute left singular vectors of the components. There should be a scatter plot for each module with a histogram along the axes for each variable"""
    n_modules = len(geometric_interaction_strength_vs_coactivation_data)
    fig = plt.figure(figsize=(8, 8 * n_modules))
    images = []
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)
    for i, (module_name, (geometric_interaction_strength, coactivation_fraction)) in enumerate(
        geometric_interaction_strength_vs_coactivation_data.items()
    ):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.scatter(
            geometric_interaction_strength.detach().cpu().numpy(),
            coactivation_fraction.detach().cpu().numpy(),
            alpha=0.2,
            s=1.2,
        )
        images.append(im)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(module_name)
        ax.set_xlabel("Geometric Interaction Strength")
        ax.set_ylabel("Coactivation Fraction")

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_geometric_interaction_strength_product_with_coactivation_fraction(
    elementwise_products: dict[str, Float[Tensor, " C C"]],
) -> Image.Image:
    """Plot elementwise products of geometric interaction strength matrices with coactivation matrices.

    This visualization shows how geometric relationships between components interact with
    their actual coactivation patterns. High values indicate strong geometric relationships
    that are also frequently coactivated.

    Args:
        elementwise_products: Dictionary mapping module names to elementwise product matrices

    Returns:
        Dictionary containing the matplotlib figure
    """
    n_modules = len(elementwise_products)
    fig = plt.figure(figsize=(8, 8 * n_modules))

    # Create GridSpec: main plots take 85% width, colorbar takes 10%, 5% gap
    gs = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)

    # Create a unified normalization for all plots
    # The range will be from -1 to 1 since geometric interaction strengths are in [-1, 1]
    # and coactivation fractions are in [0, 1], so products are in [-1, 1]
    norm = plt.Normalize(vmin=0, vmax=1)

    # Store all image objects for colorbar
    images = []

    # Iterate through modules and plot each matrix on its corresponding axis
    for i, (module_name, elementwise_product) in enumerate(elementwise_products.items()):
        # Create subplot for the main plot
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(
            elementwise_product.detach().cpu().numpy(), aspect="auto", cmap="Purples", norm=norm
        )
        images.append(im)
        ax.set_title(f"{module_name} - Geometric Interaction Strength × Coactivation Fraction")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Component index")

    # Add a unified colorbar for all plots
    if images:
        # Create colorbar in the dedicated right column
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("GIS × Coact Fraction", fontsize=12)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_single_layer_component_co_activation_fractions(
    module_name: str,
    component_co_activation_fractions: Float[Tensor, " C C"],
) -> Image.Image:
    """Plot the component co-activation fractions for a single component module."""
    fig, ax = plt.subplots(figsize=(12, 12))

    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.matshow(
        component_co_activation_fractions.detach().cpu().numpy(),
        aspect="auto",
        cmap="Blues",
        norm=norm,
    )

    ax.set_title(f"{module_name} - Component Co-Activation Fractions")
    ax.set_xlabel("Denominator Component index")
    ax.set_ylabel("Numerator Component index")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Co-activation Fraction", fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    fig.tight_layout()
    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_single_layer_component_abs_left_singular_vectors_geometric_interaction_strengths(
    module_name: str,
    geometric_interaction_strengths: Float[Tensor, " C C"],
) -> Image.Image:
    """Plot the geometric interaction strengths for a single component module."""
    fig, ax = plt.subplots(figsize=(12, 12))

    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.matshow(
        geometric_interaction_strengths.detach().cpu().numpy(),
        aspect="auto",
        cmap="Reds",
        norm=norm,
    )

    ax.set_title(f"{module_name} - Geometric Interaction Strengths")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Component index")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Geometric Interaction Strength", fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    fig.tight_layout()
    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


# def plot_single_layer_geometric_interaction_strength_vs_coactivation(
#     module_name: str,
#     geometric_interaction_strength: Float[Tensor, " C C"],
#     coactivation_fraction: Float[Tensor, " C C"],
# ) -> Image.Image:
#     """Plot the geometric interaction strength vs coactivation scatter plot for a single component module."""
#     fig, ax = plt.subplots(figsize=(12, 12))

#     ax.scatter(
#         geometric_interaction_strength.detach().cpu().numpy(),
#         coactivation_fraction.detach().cpu().numpy(),
#         alpha=0.2,
#         s=1.2,
#     )

#     ax.set_xlim(-0.01, 1.01)
#     ax.set_ylim(-0.01, 1.01)
#     ax.set_title(f"{module_name} - Geometric Interaction Strength vs Coactivation")
#     ax.set_xlabel("Geometric Interaction Strength")
#     ax.set_ylabel("Coactivation Fraction")

#     fig.tight_layout()
#     fig_img = _render_figure(fig)
#     plt.close(fig)

#     return fig_img


def plot_single_layer_geometric_interaction_strength_vs_coactivation(
    module_name: str,
    geometric_interaction_strength: Float[Tensor, " C C"],
    coactivation_fraction: Float[Tensor, " C C"],
) -> Image.Image:
    """Plot the geometric interaction strength vs coactivation scatter plot for a
    single component module with histograms of the variables along the axes."""
    # First we set up a 2 x 2 grid of subplots so that the scatter plot is in the
    # top right and the histograms are along the axes. The histograms are thin and are
    # mainly there to show the distribution of the variables.

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 7], height_ratios=[7, 1], wspace=0.1, hspace=0.1)

    # Plot the scatter plot in the top right
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.scatter(
        geometric_interaction_strength.detach().cpu().numpy(),
        coactivation_fraction.detach().cpu().numpy(),
        alpha=0.2,
        s=1.2,
    )
    ax.set_title(f"{module_name} - Geometric Interaction Strength vs Coactivation Fraction")

    # Plot the histograms along the axes, which share axes with the scatter plot. The y axis in the histograms is log scale.
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(coactivation_fraction.detach().cpu().numpy(), bins=250, orientation="horizontal")
    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel("Coactivation Fraction")
    ax.set_xlabel("Frequency")
    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(-0.01, 1.01)
    ax.hist(geometric_interaction_strength.detach().cpu().numpy(), bins=250)
    ax.set_xlabel("Geometric Interaction Strength")
    ax.set_ylabel("Frequency")

    fig_img = _render_figure(fig)
    plt.close(fig)
    return fig_img


def plot_single_layer_geometric_interaction_strength_product_with_coactivation_fraction(
    module_name: str,
    elementwise_product: Float[Tensor, " C C"],
) -> Image.Image:
    """Plot elementwise product of geometric interaction strength matrix with coactivation matrix for a single component module."""
    fig, ax = plt.subplots(figsize=(12, 12))

    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.matshow(
        elementwise_product.detach().cpu().numpy(), aspect="auto", cmap="Purples", norm=norm
    )

    ax.set_title(f"{module_name} - Geometric Interaction Strength × Coactivation Fraction")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Component index")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("GIS × Coact Fraction", fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    fig.tight_layout()
    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img
