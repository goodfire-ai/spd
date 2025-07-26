"""Plotting utilities for memorization experiments."""

from collections.abc import Mapping
from typing import Any

import torch
from matplotlib import pyplot as plt

from spd.models.component_model import ComponentModel
from spd.plotting import _plot_causal_importances_figure


def create_memorization_plot_results(
    model: ComponentModel,
    device: str | torch.device,
    dataset: Any,  # KeyValueMemorizationDataset
) -> Mapping[str, plt.Figure]:
    """Create plotting results for memorization decomposition experiments."""

    # Step 1: Get all unique keys from dataset (skip n_keys_to_plot logic)
    keys = dataset.keys.to(device)

    # Step 2: Get pre-weight activations and calculate causal importances
    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        keys, module_names=model.target_module_paths
    )[1]

    ci_raw, _ = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type="leaky_hard",
        detach_inputs=False,
    )

    # Step 3: Make the heatmap using _plot_causal_importances_figure
    fig = _plot_causal_importances_figure(
        ci_vals=ci_raw,
        title_prefix="memorization causal importances",
        colormap="Blues",
        input_magnitude=1.0,
        has_pos_dim=False,
        orientation="vertical",
    )

    return {"memorization_causal_importances": fig}
