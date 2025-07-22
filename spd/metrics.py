"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import wandb
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config
from spd.losses import calc_ce_losses
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.plotting import create_embed_ci_sample_table
from spd.utils.component_utils import calc_ci_l_zero
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


class CreateMetricsFn(Protocol):
    def __call__(
        self,
        inputs: CreateMetricsInputs,
        *args: Any,
        **kwargs: Any,
    ) -> Mapping[str, float | int | wandb.Table]: ...


def lm_kl(inputs: CreateMetricsInputs) -> Mapping[str, float | int | wandb.Table]:
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

    for fn_cfg in config.metrics_fns:
        if (fn := METRICS_FNS.get(fn_cfg.name)) is None:
            raise ValueError(f"Metric {fn_cfg.name} not found in METRICS_FNS")

        result = fn(inputs, **fn_cfg.extra_kwargs)

        if already_present_keys := set(result.keys()).intersection(metrics.keys()):
            raise ValueError(f"Metric keys {already_present_keys} already exists in metrics")

        metrics.update(result)

    return metrics


METRICS_FNS: dict[str, CreateMetricsFn] = {
    fn.__name__: fn
    for fn in [
        ci_l0,
        lm_kl,
        lm_embed,
        lm_ce_losses,
    ]
}
