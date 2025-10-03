import fnmatch
import io
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import Components
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


def plot_mean_component_cis_both_scales(
    mean_component_cis: dict[str, Float[Tensor, " C"]],
) -> tuple[Image.Image, Image.Image]:
    """
    Efficiently plot mean CI per component with both linear and log scales.

    This function optimizes the plotting by pre-processing data once and
    reusing it for both plots.

    Args:
        mean_component_cis: Dictionary mapping module names to mean CI tensors

    Returns:
        Tuple of (linear_scale_image, log_scale_image)
    """
    n_modules = len(mean_component_cis)
    max_rows = 6

    # Calculate grid dimensions once
    n_cols = (n_modules + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_modules, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 8 * n_cols
    fig_height = 3 * n_rows

    # Pre-process data once
    processed_data = []
    for module_name, mean_component_ci in mean_component_cis.items():
        sorted_components = torch.sort(mean_component_ci, descending=True)[0]
        processed_data.append((module_name, sorted_components.detach().cpu().numpy()))

    # Create both figures
    images = []
    for log_y in [False, True]:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=200)
        axs = np.array(axs)

        # Ensure axs is always 2D array for consistent indexing
        if axs.ndim == 1:
            axs = axs.reshape(n_rows, n_cols)

        # Hide unused subplots
        for i in range(n_modules, n_rows * n_cols):
            row = i % n_rows
            col = i // n_rows
            axs[row, col].set_visible(False)

        for i, (module_name, sorted_components_np) in enumerate(processed_data):
            # Calculate position in grid (fill column by column)
            row = i % n_rows
            col = i // n_rows
            ax = axs[row, col]

            if log_y:
                ax.set_yscale("log")

            ax.scatter(
                range(len(sorted_components_np)),
                sorted_components_np,
                marker="x",
                s=10,
            )

            # Only add x-label to bottom row of each column
            if row == n_rows - 1 or i == n_modules - 1:
                ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(module_name, fontsize=10)

        fig.tight_layout()
        img = _render_figure(fig)
        plt.close(fig)
        images.append(img)

    return images[0], images[1]


def get_single_feature_causal_importances(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    sampling: Literal["continuous", "binomial"],
    sigmoid_type: SigmoidTypes,
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

    pre_weight_acts = model(batch, mode="input_cache", module_names=list(model.components.keys()))[
        1
    ]

    ci_raw, ci_upper_leaky_raw = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type=sigmoid_type,
        detach_inputs=False,
        sampling=sampling,
    )

    return ci_raw, ci_upper_leaky_raw


def get_single_feature_pre_sigmoid_gate_outputs(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
) -> dict[str, Float[Tensor, "batch C"]]:
    """Compute pre-sigmoid gate output arrays for single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features

    Returns:
        Dictionary of pre-sigmoid gate output arrays (2D tensors)
    """
    # Create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only use the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model(batch, mode="input_cache", module_names=list(model.components.keys()))[
        1
    ]

    pre_sigmoid_outputs = model.calc_pre_sigmoid_gate_outputs(
        pre_weight_acts=pre_weight_acts,
        detach_inputs=False,
    )

    return pre_sigmoid_outputs


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    sampling: Literal["continuous", "binomial"],
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
        sampling=sampling,
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
    """Plot the activation density of each component as a histogram in a grid layout."""

    n_modules = len(component_activation_density)
    max_rows = 6

    # Calculate grid dimensions
    n_cols = (n_modules + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_modules, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 5 * n_cols
    fig_height = 5 * n_rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axs = np.array(axs)

    # Ensure axs is always 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    # Hide unused subplots
    for i in range(n_modules, n_rows * n_cols):
        row = i % n_rows
        col = i // n_rows
        axs[row, col].set_visible(False)

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, density) in enumerate(component_activation_density.items()):
        # Calculate position in grid (fill column by column)
        row = i % n_rows
        col = i // n_rows
        ax = axs[row, col]

        data = density.detach().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(module_name)  # Add module name as title to each subplot

        # Only add x-label to bottom row of each column
        if row == n_rows - 1 or i == n_modules - 1:
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
    """Plot histograms of mask values for all layers in a grid layout.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Single figure with subplots for each layer.
    """
    n_layers = len(causal_importances)
    max_rows = 6

    # Calculate grid dimensions
    n_cols = (n_layers + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_layers, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axs = np.array(axs)

    # Ensure axs is always 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    # Hide unused subplots
    for i in range(n_layers, n_rows * n_cols):
        row = i % n_rows
        col = i // n_rows
        axs[row, col].set_visible(False)

    for i, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
        layer_name = layer_name_raw.replace(".", "_")

        # Calculate position in grid (fill column by column)
        row = i % n_rows
        col = i // n_rows
        ax = axs[row, col]

        data = layer_ci.flatten().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(f"Causal importances for {layer_name}")

        # Only add x-label to bottom row of each column
        if row == n_rows - 1 or i == n_layers - 1:
            ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img
def plot_pre_sigmoid_gate_outputs(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    identity_patterns: list[str] | None = None,
    dense_patterns: list[str] | None = None,
    title_formatter: Callable[[str], str] | None = None,
) -> tuple[dict[str, Image.Image], dict[str, Float[Tensor, " C"]]]:
    """Plot the pre-sigmoid gate output values for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features
        identity_patterns: List of patterns to match for identity permutation
        dense_patterns: List of patterns to match for dense permutation
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        Tuple of:
            - Dictionary of figures with key 'pre_sigmoid_gate_outputs'
            - Dictionary of permutation indices for pre-sigmoid gate outputs
    """
    # Get the pre-sigmoid gate output arrays
    pre_sigmoid_raw = get_single_feature_pre_sigmoid_gate_outputs(
        model=model,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=input_magnitude,
    )

    pre_sigmoid: dict[str, Float[Tensor, "... C"]] = {}
    all_perm_indices: dict[str, Float[Tensor, " C"]] = {}
    for k in pre_sigmoid_raw:
        # Determine permutation strategy based on patterns
        if identity_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in identity_patterns):
            pre_sigmoid[k], all_perm_indices[k] = permute_to_identity(ci_vals=pre_sigmoid_raw[k])
        elif dense_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in dense_patterns):
            pre_sigmoid[k], all_perm_indices[k] = permute_to_dense(ci_vals=pre_sigmoid_raw[k])
        else:
            pre_sigmoid[k] = pre_sigmoid_raw[k]

    # Determine if we have position dimension
    has_pos_dim = len(batch_shape) == 3

    # Create the plot
    fig_img = _plot_causal_importances_figure(
        ci_vals=pre_sigmoid,
        title_prefix="Pre-sigmoid gate outputs",
        colormap="RdBu_r",  # Use a different colormap to distinguish from post-sigmoid
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        title_formatter=title_formatter,
    )

    return {"pre_sigmoid_gate_outputs": fig_img}, all_perm_indices


def plot_component_weight_heatmaps(
    model: ComponentModel,
    embedding_module_name: str | None = None,
    unembedding_module_name: str | None = None,
    figsize_per_component: tuple[float, float] = (4, 3),
    dpi: int = 150,
    input_magnitude: float = 0.75,
) -> dict[str, Image.Image]:
    """Plot heatmaps of component weights and their interactions with embedding/unembedding.
    Only plots components that actually respond to inputs (have non-zero causal importance).
    
    Args:
        model: The ComponentModel
        embedding_module_name: Name of the embedding module in the target model (e.g., 'token_embed')
        unembedding_module_name: Name of the unembedding module in the target model (e.g., 'unembed')
        figsize_per_component: Figure size for each component heatmap
        dpi: DPI for the plots
        input_magnitude: Magnitude of input features for causal importance calculation
        
    Returns:
        Dictionary of figures with keys for each layer's component heatmaps
    """
    figures = {}
    
    # Get embedding and unembedding weights if specified
    embedding_weight = None
    unembedding_weight = None
    
    if embedding_module_name:
        try:
            # Try to get as a module first
            embedding_weight = model.target_weight(embedding_module_name)
        except Exception:
            try:
                # If that fails, try to get as a parameter
                embedding_weight = dict(model.target_model.named_parameters())[embedding_module_name]
            except Exception as e:
                print(f"Warning: Could not load embedding weight from {embedding_module_name}: {e}")
                embedding_weight = None
    
    if unembedding_module_name:
        try:
            # Try to get as a module first
            unembedding_weight = model.target_weight(unembedding_module_name)
        except Exception:
            try:
                # If that fails, try to get as a parameter
                unembedding_weight = dict(model.target_model.named_parameters())[unembedding_module_name]
            except Exception as e:
                print(f"Warning: Could not load unembedding weight from {unembedding_module_name}: {e}")
                unembedding_weight = None
    
    # First, calculate causal importances to identify active components
    print("Calculating causal importances to identify active components...")
    device = next(iter(model.parameters())).device
    
    # Create a batch of inputs with single active features
    # For this model, we need to use the embedding dimension, not the component input dimension
    # The model expects inputs with the same dimension as the embedding
    if hasattr(model.target_model, 'W_E'):
        n_features = model.target_model.W_E.shape[0]  # Use embedding dimension
    else:
        # Fallback to component input dimension
        first_component = next(iter(model.components.values()))
        n_features = first_component.V.shape[0]  # d_in
    
    print(f"Using input dimension: {n_features}")
    batch = torch.eye(n_features, device=device) * input_magnitude
    
    # Get pre-weight activations
    _, pre_weight_acts = model(batch, mode="input_cache", module_names=list(model.components.keys()))
    
    # Calculate causal importances
    ci, ci_upper_leaky = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type=model.config.sigmoid_type if hasattr(model, 'config') else 'leaky_hard',
        sampling='continuous',
        detach_inputs=False,
    )
    
    # Collect all weight matrices to calculate global min/max
    all_weight_matrices = []
    all_embedding_matrices = []
    all_unembedding_matrices = []
    
    # Process each layer
    for layer_name, component in model.components.items():
        C = component.C
        V = component.V  # Shape: (d_in, C)
        U = component.U  # Shape: (C, d_out)
        
        # Get causal importance for this layer
        ci_layer = ci_upper_leaky[layer_name]  # Shape: (n_features, C)
        
        # Find active components (those with non-zero causal importance)
        active_components = []
        generalist_components = []  # Components that respond to >50% of input dimensions
        
        for c in range(C):
            # Check if this component responds to any input
            if ci_layer[:, c].abs().max() > 1e-6:  # Threshold for "active"
                active_components.append(c)
                
                # Check if this component responds to more than 50% of input dimensions
                n_responsive_inputs = (ci_layer[:, c].abs() > 1e-6).sum().item()
                total_inputs = ci_layer.shape[0]
                if n_responsive_inputs > 0.5 * total_inputs:
                    generalist_components.append(c)
        
        print(f"Layer {layer_name}: {len(active_components)}/{C} components are active")
        if generalist_components:
            print(f"  Generalist components (respond to >50% of inputs): {generalist_components}")
        
        if not active_components:
            print(f"  Skipping layer {layer_name} - no active components")
            continue
        
        # Collect weight matrices for global min/max calculation
        for c in active_components:
            v_c = V[:, c]  # Shape: (d_in,)
            u_c = U[c, :]  # Shape: (d_out,)
            rank_one_matrix = torch.outer(v_c, u_c)  # Shape: (d_in, d_out)
            all_weight_matrices.append(rank_one_matrix)
        
        # Collect embedding interaction matrices (only for mlp_in layers)
        if embedding_weight is not None and "mlp_in" in layer_name:
            # W_in^c @ W_E: Take outer product U[c] @ V[c] to get component matrix, then multiply with W_E
            for c in active_components:
                v_c = V[:, c]  # Shape: (d_in,)
                u_c = U[c, :]  # Shape: (d_out,)
                # Create component matrix: U[c] @ V[c] = (d_out, d_in)
                component_matrix = torch.outer(u_c, v_c)  # Shape: (d_out, d_in) = (25, 1000)
                # Multiply with embedding: (d_out, d_in) @ (d_in, n_inputs) -> (d_out, n_inputs)
                embedding_interaction = torch.matmul(component_matrix, embedding_weight.T)  # Shape: (25, 100)
                all_embedding_matrices.append(embedding_interaction)
        
        # Collect unembedding interaction matrices (only for mlp_out layers)
        if unembedding_weight is not None and "mlp_out" in layer_name:
            # W_U @ W_out^c: Take outer product U[c] @ V[c] to get component matrix, then multiply with W_U
            for c in active_components:
                v_c = V[:, c]  # Shape: (d_in,)
                u_c = U[c, :]  # Shape: (d_out,)
                # Create component matrix: U[c] @ V[c] = (d_out, d_in)
                component_matrix = torch.outer(u_c, v_c)  # Shape: (d_out, d_in) = (1000, 25)
                # Multiply with unembedding: (n_outputs, d_out) @ (d_out, d_in) -> (n_outputs, d_in)
                unembedding_interaction = torch.matmul(unembedding_weight.T, component_matrix)  # Shape: (100, 25)
                all_unembedding_matrices.append(unembedding_interaction)
    
    # Calculate global min/max for fixed color scale
    if all_weight_matrices:
        global_min = min(tensor.min().item() for tensor in all_weight_matrices)
        global_max = max(tensor.max().item() for tensor in all_weight_matrices)
        # Ensure 0 is in the center
        abs_max = max(abs(global_min), abs(global_max))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    
    if all_embedding_matrices:
        embedding_min = min(tensor.min().item() for tensor in all_embedding_matrices)
        embedding_max = max(tensor.max().item() for tensor in all_embedding_matrices)
        embedding_abs_max = max(abs(embedding_min), abs(embedding_max))
        embedding_vmin, embedding_vmax = -embedding_abs_max, embedding_abs_max
    else:
        embedding_vmin, embedding_vmax = -1, 1
    
    if all_unembedding_matrices:
        unembedding_min = min(tensor.min().item() for tensor in all_unembedding_matrices)
        unembedding_max = max(tensor.max().item() for tensor in all_unembedding_matrices)
        unembedding_abs_max = max(abs(unembedding_min), abs(unembedding_max))
        unembedding_vmin, unembedding_vmax = -unembedding_abs_max, unembedding_abs_max
    else:
        unembedding_vmin, unembedding_vmax = -1, 1
    
    print(f"Global color scale - weights: [{vmin:.3f}, {vmax:.3f}], embedding: [{embedding_vmin:.3f}, {embedding_vmax:.3f}], unembedding: [{unembedding_vmin:.3f}, {unembedding_vmax:.3f}]")
    
    # Now create the plots
    for layer_name, component in model.components.items():
        C = component.C
        V = component.V  # Shape: (d_in, C)
        U = component.U  # Shape: (C, d_out)
        
        # Get causal importance for this layer
        ci_layer = ci_upper_leaky[layer_name]  # Shape: (n_features, C)
        
        # Find active components and generalist components
        active_components = []
        generalist_components = []
        
        for c in range(C):
            if ci_layer[:, c].abs().max() > 1e-6:
                active_components.append(c)
                
                # Check if this component responds to more than 50% of input dimensions
                n_responsive_inputs = (ci_layer[:, c].abs() > 1e-6).sum().item()
                total_inputs = ci_layer.shape[0]
                if n_responsive_inputs > 0.5 * total_inputs:
                    generalist_components.append(c)
        
        if not active_components:
            continue
        
        # Create subplots for this layer (only active components)
        n_active = len(active_components)
        n_cols = min(4, n_active)  # Max 4 components per row
        n_rows = (n_active + n_cols - 1) // n_cols  # Ceiling division
        
        fig_width = n_cols * figsize_per_component[0]
        fig_height = n_rows * figsize_per_component[1]
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            squeeze=False
        )
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, c in enumerate(active_components):
            ax = axes_flat[i]
            
            # Get the c-th component
            v_c = V[:, c]  # Shape: (d_in,)
            u_c = U[c, :]  # Shape: (d_out,)
            
            # Create rank-one matrix for this component
            rank_one_matrix = torch.outer(v_c, u_c)  # Shape: (d_in, d_out)
            
            
            # Plot the heatmap with fixed color scale
            im = ax.matshow(
                rank_one_matrix.detach().cpu().numpy(),
                cmap='bwr',
                aspect='auto',
                vmin=vmin,
                vmax=vmax
            )
            
            # Create title with special marking for generalist components
            if c in generalist_components:
                n_responsive = (ci_layer[:, c].abs() > 1e-6).sum().item()
                ax.set_title(f'Component {c} (GENERALIST: {n_responsive}/{total_inputs} inputs)', 
                           color='red', fontweight='bold')
            else:
                ax.set_title(f'Component {c}')
            ax.set_xlabel('Output dim')
            ax.set_ylabel('Input dim')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(n_active, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        fig_img = _render_figure(fig)
        plt.close(fig)
        
        figures[f"component_weights_{layer_name}"] = fig_img
        
        # Plot W_in^c @ W_embedding if embedding is available (only for mlp_in layers)
        if embedding_weight is not None and "mlp_in" in layer_name and all_embedding_matrices:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(fig_width, fig_height),
                dpi=dpi,
                squeeze=False
            )
            axes_flat = axes.flatten()
            
            for i, c in enumerate(active_components):
                ax = axes_flat[i]
                
                # Get the embedding interaction matrix for this component
                v_c = V[:, c]  # Shape: (d_in,)
                u_c = U[c, :]  # Shape: (d_out,)
                component_matrix = torch.outer(u_c, v_c)  # Shape: (d_out, d_in) = (25, 1000)
                embedding_interaction = torch.matmul(component_matrix, embedding_weight.T)  # Shape: (25, 100)
                
                # Plot the 2D heatmap (d_out x n_inputs) with fixed color scale
                im = ax.matshow(
                    embedding_interaction.detach().cpu().numpy(),
                    cmap='bwr',
                    aspect='auto',
                    vmin=embedding_vmin,
                    vmax=embedding_vmax
                )
                
                # Create title with special marking for generalist components
                if c in generalist_components:
                    n_responsive = (ci_layer[:, c].abs() > 1e-6).sum().item()
                    ax.set_title(f'W_in^{c} @ W_embedding (GENERALIST: {n_responsive}/{total_inputs} inputs)', 
                               color='red', fontweight='bold')
                else:
                    ax.set_title(f'W_in^{c} @ W_embedding')
                ax.set_xlabel('Input dim')
                ax.set_ylabel('Hidden neurons')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Hide unused subplots
            for i in range(n_active, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            fig_img = _render_figure(fig)
            plt.close(fig)
            
            figures[f"input_embedding_interaction_{layer_name}"] = fig_img
        
        # Plot W_unembedding @ W_out^c if unembedding is available (only for mlp_out layers)
        if unembedding_weight is not None and "mlp_out" in layer_name and all_unembedding_matrices:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(fig_width, fig_height),
                dpi=dpi,
                squeeze=False
            )
            axes_flat = axes.flatten()
            
            for i, c in enumerate(active_components):
                ax = axes_flat[i]
                
                # Get the unembedding interaction matrix for this component
                v_c = V[:, c]  # Shape: (d_in,)
                u_c = U[c, :]  # Shape: (d_out,)
                component_matrix = torch.outer(u_c, v_c)  # Shape: (d_out, d_in) = (1000, 25)
                unembedding_interaction = torch.matmul(unembedding_weight.T, component_matrix)  # Shape: (100, 25)
                
                # Plot the 2D heatmap (n_outputs x d_in) with fixed color scale
                im = ax.matshow(
                    unembedding_interaction.detach().cpu().numpy(),
                    cmap='bwr',
                    aspect='auto',
                    vmin=unembedding_vmin,
                    vmax=unembedding_vmax
                )
                
                # Create title with special marking for generalist components
                if c in generalist_components:
                    n_responsive = (ci_layer[:, c].abs() > 1e-6).sum().item()
                    ax.set_title(f'W_unembedding @ W_out^{c} (GENERALIST: {n_responsive}/{total_inputs} inputs)', 
                               color='red', fontweight='bold')
                else:
                    ax.set_title(f'W_unembedding @ W_out^{c}')
                ax.set_xlabel('Input dim')
                ax.set_ylabel('Output dim')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Hide unused subplots
            for i in range(n_active, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            fig_img = _render_figure(fig)
            plt.close(fig)
            
            figures[f"output_unembedding_interaction_{layer_name}"] = fig_img
    
    return figures
