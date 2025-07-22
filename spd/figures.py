"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.plotting import (
    plot_causal_importance_vals,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
    plot_UV_matrices,
)
from spd.utils.component_utils import component_activation_statistics


@dataclass
class CreateFiguresInputs:
    model: ComponentModel
    components: dict[str, LinearComponent | EmbeddingComponent]
    gates: dict[str, GateMLP | VectorGateMLP]
    causal_importances: dict[str, Float[Tensor, "... C"]]
    target_out: Float[Tensor, "... d_model_out"]
    batch: Int[Tensor, "... d_model_in"] | Float[Tensor, "... d_model_in"]
    device: str | torch.device
    config: Config
    step: int
    eval_loader: (
        DataLoader[Int[Tensor, "..."]]
        | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]
    )
    n_eval_steps: int


class CreateFiguresFn(Protocol):
    def __call__(
        self,
        inputs: CreateFiguresInputs,
        *args: Any,
        **kwargs: Any,
    ) -> Mapping[str, plt.Figure]: ...


def ci_histograms(inputs: CreateFiguresInputs) -> Mapping[str, plt.Figure]:
    return plot_ci_histograms(causal_importances=inputs.causal_importances)


def mean_component_activation_counts(inputs: CreateFiguresInputs) -> Mapping[str, plt.Figure]:
    mean_component_activation_counts = component_activation_statistics(
        model=inputs.model,
        dataloader=inputs.eval_loader,
        n_steps=inputs.n_eval_steps,
        device=str(inputs.device),
    )[1]
    return {
        "mean_component_activation_counts": plot_mean_component_activation_counts(
            mean_component_activation_counts=mean_component_activation_counts,
        )
    }


def uv_and_identity_ci(inputs: CreateFiguresInputs) -> Mapping[str, plt.Figure]:
    figures, all_perm_indices = plot_causal_importance_vals(
        model=inputs.model,
        components=inputs.components,
        gates=inputs.gates,
        batch_shape=inputs.batch.shape,
        device=inputs.device,
        input_magnitude=0.75,
        sigmoid_type=inputs.config.sigmoid_type,
    )

    uv_matrices = plot_UV_matrices(components=inputs.components, all_perm_indices=all_perm_indices)

    return {
        **figures,
        "UV_matrices": uv_matrices,
    }


def create_figures(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, GateMLP | VectorGateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Int[Tensor, "... d_model_in"] | Float[Tensor, "... d_model_in"],
    device: str | torch.device,
    config: Config,
    step: int,
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
) -> Mapping[str, plt.Figure]:
    """Create figures for logging.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        causal_importances: Current causal importances
        target_out: Output of target model
        batch: Current batch tensor
        device: Current device (cuda/cpu)
        config: The full configuration object
        step: Current training step
        eval_loader: Evaluation loader
        n_eval_steps: Number of evaluation steps

    Returns:
        Dictionary of figures
    """

    fig_dict = {}
    inputs = CreateFiguresInputs(
        model=model,
        components=components,
        gates=gates,
        causal_importances=causal_importances,
        target_out=target_out,
        batch=batch,
        device=device,
        config=config,
        step=step,
        eval_loader=eval_loader,
        n_eval_steps=n_eval_steps,
    )
    for fn_cfg in config.figures_fns:
        if (fn := FIGURES_FNS.get(fn_cfg.name)) is None:
            raise ValueError(f"Figure {fn_cfg.name} not found in FIGURES_FNS")

        result = fn(inputs, **fn_cfg.extra_kwargs)

        if already_present_keys := set(result.keys()).intersection(fig_dict.keys()):
            raise ValueError(f"Figure keys {already_present_keys} already exists in fig_dict")

        fig_dict.update(result)

    return fig_dict


FIGURES_FNS: dict[str, CreateFiguresFn] = {
    fn.__name__: fn
    for fn in [
        ci_histograms,
        mean_component_activation_counts,
        uv_and_identity_ci,
    ]
}
