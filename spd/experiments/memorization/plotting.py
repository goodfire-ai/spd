"""Plotting utilities for memorization experiments."""

import torch
from matplotlib import pyplot as plt

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.plotting import (
    _plot_causal_importances_figure,
    plot_UV_matrices,
)
from spd.utils.component_utils import calc_causal_importances
from spd.utils.target_ci_solutions import permute_to_identity_hungarian


def create_memorization_plot_results(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, GateMLP | VectorGateMLP],
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    dataset,  # KeyValueMemorizationDataset # type: ignore
    return_raw_cis: bool = False,
    n_keys_to_plot: int | None = None,
    **_,
) -> dict[str, plt.Figure]:
    """Create plotting results for memorization decomposition experiments.

    Similar to create_toy_model_plot_results, but uses actual keys from the dataset
    instead of identity inputs.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        batch_shape: Shape of the batch
        device: Device to use
        dataset: KeyValueMemorizationDataset containing the keys
        return_raw_cis: Whether to include raw permuted CIs in the results
        n_keys_to_plot: Number of keys to plot (defaults to min(n_pairs, d_model))
        **_: Additional keyword arguments (ignored)

    Returns:
        Dictionary of figures (and raw CIs if return_raw_cis=True)
    """
    fig_dict = {}

    # Determine how many keys to plot
    n_pairs = dataset.n_pairs
    n_keys_to_plot = n_pairs if n_keys_to_plot is None else min(n_keys_to_plot, n_pairs)

    # Get a batch of keys from the dataset
    # We'll use the first n_keys_to_plot keys for visualization
    keys = dataset.keys[:n_keys_to_plot].to(device)

    # Forward pass through the model to get pre-weight activations
    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        keys, module_names=list(components.keys())
    )[1]
    Vs = {module_name: v.V for module_name, v in components.items()}

    # Calculate causal importances
    ci_raw, ci_upper_leaky_raw = calc_causal_importances(
        pre_weight_acts=pre_weight_acts, Vs=Vs, gates=gates, detach_inputs=False
    )

    # Permute to make it closer to identity for visualization
    ci = {}
    ci_upper_leaky = {}
    all_perm_indices = {}

    for k in ci_raw:
        ci[k], _ = permute_to_identity_hungarian(ci_vals=ci_raw[k])
        ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity_hungarian(
            ci_vals=ci_upper_leaky_raw[k]
        )

    # Create figures
    figures = {}

    # Plot raw causal importances (blue)
    ci_fig = _plot_causal_importances_figure(
        ci_vals=ci,
        title_prefix="importance values lower leaky relu",
        colormap="Blues",
        input_magnitude=1.0,  # Keys are unit norm
        has_pos_dim=False,
        orientation="vertical",
        title_formatter=None,
    )
    figures["causal_importances"] = ci_fig

    # Plot upper leaky causal importances (red)
    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=1.0,  # Keys are unit norm
        has_pos_dim=False,
        orientation="vertical",
        title_formatter=None,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    # Add raw permuted CIs if requested
    if return_raw_cis:
        figures["raw_ci_permuted"] = ci
        figures["raw_ci_upper_leaky_permuted"] = ci_upper_leaky

    fig_dict.update(figures)

    # Plot UV matrices
    fig_dict["UV_matrices"] = plot_UV_matrices(
        components=components, all_perm_indices=all_perm_indices
    )

    # Add a figure showing which keys were used for plotting
    # fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=300)
    # keys_np = keys.detach().cpu().numpy()
    # im = ax.matshow(keys_np, aspect="auto", cmap="viridis")
    # ax.set_xlabel("Key dimension")
    # ax.set_ylabel("Key index")
    # ax.set_title(f"Keys used for causal importance visualization (first {n_keys_to_plot} keys)")
    # ax.xaxis.tick_bottom()
    # ax.xaxis.set_label_position("bottom")
    # fig.colorbar(im, ax=ax)
    # fig_dict["keys_used"] = fig

    return fig_dict
