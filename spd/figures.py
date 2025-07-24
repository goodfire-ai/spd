"""Figures for SPD experiments.

This file contains visualizations that can be logged during SPD optimization.
These can be selected and configured in the config file.
"""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from einops import reduce
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import (
    plot_causal_importance_vals,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
    plot_UV_matrices,
)
from spd.utils.general_utils import extract_batch_data


@dataclass
class FigureInput:
    ci: dict[str, Float[Tensor, "... C"]]
    batch: Int[Tensor, "..."] | Float[Tensor, "..."]


class CreateFiguresFn(Protocol):
    def __call__(
        self,
        model: ComponentModel,
        config: Config,
        input_batches: list[FigureInput],
        *args: Any,
        **kwargs: Any,
    ) -> Mapping[str, plt.Figure]: ...


def ci_histograms(
    model: ComponentModel,  # pyright: ignore[reportUnusedParameter]
    config: Config,  # pyright: ignore[reportUnusedParameter]
    input_batches: list[FigureInput],
) -> Mapping[str, plt.Figure]:
    """Create CI histogram figures."""
    causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    for inputs in input_batches:
        for k, v in inputs.ci.items():
            causal_importances[k].append(v)

    combined_causal_importances = {k: torch.cat(v) for k, v in causal_importances.items()}
    fig = plot_ci_histograms(causal_importances=combined_causal_importances)
    return {"causal_importances_hist": fig}


def mean_component_activation_counts(
    model: ComponentModel,
    config: Config,
    input_batches: list[FigureInput],
) -> Mapping[str, plt.Figure]:
    """Create mean component activation counts figure."""
    n_tokens = 0
    device = next(iter(model.parameters())).device

    component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: torch.zeros(model.C, device=device) for module_name in model.components
    }

    for inputs in input_batches:
        batch_n_tokens = next(iter(inputs.ci.values())).shape[:-1].numel()
        n_tokens += batch_n_tokens

        for module_name, ci_vals in inputs.ci.items():
            active_components = ci_vals > config.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            component_activation_counts[module_name] += n_activations_per_component

    mean_counts_per_module = {
        module_name: component_activation_counts[module_name] / n_tokens
        for module_name in model.components
    }
    fig = plot_mean_component_activation_counts(mean_counts_per_module)
    return {"mean_component_activation_counts": fig}


def uv_and_identity_ci(
    model: ComponentModel,
    config: Config,
    input_batches: list[FigureInput],
) -> Mapping[str, plt.Figure]:
    """Create UV and identity CI figures."""
    device = next(iter(model.parameters())).device

    # Just need the batch shape from any input
    if not input_batches:
        raise ValueError("No input batches provided")

    batch_shape = input_batches[0].batch.shape

    figures, all_perm_indices = plot_causal_importance_vals(
        model=model,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=0.75,
        sigmoid_type=config.sigmoid_type,
    )

    uv_matrices_fig = plot_UV_matrices(
        components=model.components, all_perm_indices=all_perm_indices
    )

    return {
        **figures,
        "uv_matrices": uv_matrices_fig,
    }


FIGURES_FNS: dict[str, CreateFiguresFn] = {
    fn.__name__: fn
    for fn in [
        ci_histograms,
        mean_component_activation_counts,
        uv_and_identity_ci,
    ]
}


def create_figures(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str | torch.device,
    config: Config,
    n_eval_steps: int,
) -> Mapping[str, plt.Figure]:
    """Create figures for logging."""
    # Collect all inputs first
    inputs_list: list[FigureInput] = []

    # TODO(oli): potentially move all inputs onto cpu, then move to gpu inside functions
    for _ in range(n_eval_steps):
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)
        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )
        ci, _ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=config.sigmoid_type
        )

        inputs_list.append(FigureInput(ci=ci, batch=batch))

    # Process figures
    out: dict[str, plt.Figure] = {}

    for fn_cfg in config.figures_fns:
        if (fn := FIGURES_FNS.get(fn_cfg.name)) is None:
            raise ValueError(f"Figure function {fn_cfg.name} not found in FIGURES_FNS")

        result = fn(model, config, inputs_list, **fn_cfg.extra_kwargs)

        if already_present_keys := set(result.keys()).intersection(out.keys()):
            raise ValueError(f"Figure keys {already_present_keys} already exists in figures")

        out.update(result)

    return out
