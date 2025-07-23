"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import wandb
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import Config
from spd.losses import calc_ce_losses
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.plotting import (
    create_embed_ci_sample_table,
    plot_causal_importance_vals,
    plot_ci_histograms,
    plot_cosine_similarity_histograms,
    plot_mean_component_activation_counts,
    plot_UV_matrices,
)
from spd.utils.component_utils import (
    calc_ci_l_zero,
    calc_component_target_cosine_similarities,
    component_activation_statistics,
)
from spd.utils.general_utils import calc_kl_divergence_lm


@dataclass
class CreateMetricsInputs:
    model: ComponentModel
    components: dict[str, LinearComponent | EmbeddingComponent]
    gates: dict[str, GateMLP | VectorGateMLP]
    causal_importances: dict[str, Float[Tensor, "... C"]]
    target_out: Float[Tensor, "... d_model_out"]
    unmasked_component_out: Float[Tensor, "... d_model_out"]
    masked_component_out: Float[Tensor, "... d_model_out"]
    batch: Tensor
    device: str
    config: Config
    step: int


def lm_kl(inputs: CreateMetricsInputs):
    kl_vs_target = calc_kl_divergence_lm(
        pred=inputs.unmasked_component_out, target=inputs.target_out
    )
    kl_vs_target_masked = calc_kl_divergence_lm(
        pred=inputs.masked_component_out, target=inputs.target_out
    )

    return {
        "misc/unmasked_kl_loss_vs_target": kl_vs_target.item(),
        "misc/masked_kl_loss_vs_target": kl_vs_target_masked.item(),
    }


def lm_ce_losses(inputs: CreateMetricsInputs) -> Mapping[str, float | int | wandb.Table]:
    return calc_ce_losses(
        model=inputs.model,
        batch=inputs.batch,
        components=inputs.components,
        masks=inputs.causal_importances,
        unmasked_component_logits=inputs.unmasked_component_out,
        masked_component_logits=inputs.masked_component_out,
        target_logits=inputs.target_out,
    )


def lm_embed(inputs: CreateMetricsInputs) -> Mapping[str, float | int | wandb.Table]:
    for key in ["transformer.wte", "model.embed_tokens"]:
        if key in inputs.causal_importances:
            embed_ci_table = create_embed_ci_sample_table(inputs.causal_importances, key)
            return {"misc/embed_ci_sample": embed_ci_table}
    raise ValueError("No embedding components found in causal importances")


def ci_l0(inputs: CreateMetricsInputs) -> Mapping[str, float | int | wandb.Table]:
    l0_metrics = {}
    ci_l_zero = calc_ci_l_zero(causal_importances=inputs.causal_importances)
    for layer_name, layer_ci_l_zero in ci_l_zero.items():
        l0_metrics[f"{layer_name}/ci_l0"] = layer_ci_l_zero

    return l0_metrics


METRICS_FNS = [
    ci_l0,
    lm_kl,
    lm_embed,
    lm_ce_losses,
]


def create_metrics(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, GateMLP | VectorGateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Tensor,
    device: str,
    config: Config,
    step: int,
) -> Mapping[str, float | int | wandb.Table]:
    """Create metrics for logging."""
    metrics: dict[str, float | int | wandb.Table] = {"misc/step": step}

    masked_component_out = model.forward_with_components(
        batch, components=components, masks=causal_importances
    )
    unmasked_component_out = model.forward_with_components(batch, components=components, masks=None)

    inputs = CreateMetricsInputs(
        model=model,
        components=components,
        gates=gates,
        causal_importances=causal_importances,
        target_out=target_out,
        unmasked_component_out=unmasked_component_out,
        masked_component_out=masked_component_out,
        batch=batch,
        device=device,
        config=config,
        step=step,
    )

    for fn in METRICS_FNS:
        if fn.__name__ not in config.metrics_fns:
            continue

        if set(fn(inputs).keys()) & set(metrics.keys()):
            raise ValueError(f"Metric {fn.__name__} already exists in metrics")

        metrics.update(fn(inputs))

    return metrics


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


def cosine_similarity_histograms(inputs: CreateFiguresInputs) -> Mapping[str, plt.Figure]:
    """Create cosine similarity histograms for components vs target model."""
    # Get the target model from the component model
    target_model = inputs.model.model

    # Calculate cosine similarities
    cosine_histograms = calc_component_target_cosine_similarities(
        components=inputs.components,
        target_model=target_model,
    )

    # Create plots
    return plot_cosine_similarity_histograms(cosine_histograms=cosine_histograms)


FIGURES_FNS: list[Callable[[CreateFiguresInputs], Mapping[str, plt.Figure]]] = [
    ci_histograms,
    mean_component_activation_counts,
    uv_and_identity_ci,
    cosine_similarity_histograms,
]


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
    for fn in FIGURES_FNS:
        if fn.__name__ not in config.figures_fns:
            continue

        if set(fn(inputs).keys()) & set(fig_dict.keys()):
            raise ValueError(f"Figure {fn.__name__} already exists in fig_dict")

        fig_dict.update(fn(inputs))

    return fig_dict
