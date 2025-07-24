"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import einops
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data


@dataclass
class MetricsBatchInputs:
    batch: Tensor
    target_out: Float[Tensor, "... vocab"]
    ci: dict[str, Float[Tensor, "... C"]]


class CreateMetricsFn(Protocol):
    def __call__(
        self,
        config: Config,
        input_batches: list[MetricsBatchInputs],
        *args: Any,
        **kwargs: Any,
    ) -> Mapping[str, float | int | wandb.Table]: ...


def l0(config: Config, input_batches: list[MetricsBatchInputs]) -> Mapping[str, float]:
    all_l0s = defaultdict[str, list[float]](list)
    for input in input_batches:
        for layer_name, ci in input.ci.items():
            l0 = (ci > config.ci_alive_threshold).float().sum(-1).mean().item()
            all_l0s[layer_name].append(l0)

    out = {}
    for layer_name, l0s in all_l0s.items():
        out[f"l0/{layer_name}"] = sum(l0s) / len(l0s)

    return out


def ce_kl(
    config: Config, input_batches: list[MetricsBatchInputs], *, model: ComponentModel
) -> Mapping[str, float]:
    ce_losses = defaultdict[str, list[float]](list)
    for input in input_batches:
        for key, value in _calc_ce_and_kl_losses(input, model, config.ci_alive_threshold).items():
            ce_losses[key].append(value)
    return {k: sum(v) / len(v) for k, v in ce_losses.items()}


def _calc_ce_and_kl_losses(
    inputs: MetricsBatchInputs,
    model: ComponentModel,
    rounding_threshold: float,
) -> Mapping[str, float]:
    ci = inputs.ci
    target_out = inputs.target_out
    batch = inputs.batch

    assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

    # make sure labels don't "wrap around": you **can't** predict the first token.
    masked_batch = batch.clone()
    masked_batch[:, 0] = -100  # F.cross_entropy ignores -99
    flat_masked_batch = masked_batch.flatten()

    def ce_vs_labels(logits: Tensor) -> float:
        flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
        return F.cross_entropy(flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100).item()

    def kl_vs_target(logits: Tensor) -> float:
        return calc_kl_divergence_lm(pred=logits, target=target_out).item()

    # CE When...
    # we use the causal importances as a mask
    ci_masked_logits = model.forward_with_components(batch, masks=ci)
    ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
    ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

    # we use the regular stochastic masks
    stoch_masks = calc_stochastic_masks(ci, n_mask_samples=1)[0]
    stoch_masked_logits = model.forward_with_components(batch, masks=stoch_masks)
    stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
    stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

    # we use all components
    nonmask = {k: torch.ones_like(v) for k, v in ci.items()}
    unmasked_logits = model.forward_with_components(batch, masks=nonmask)
    unmasked_ce_loss = ce_vs_labels(unmasked_logits)
    unmasked_kl_loss = kl_vs_target(unmasked_logits)

    # we use completely random masks
    random_mask = {k: torch.rand_like(v) for k, v in ci.items()}
    random_masked_logits = model.forward_with_components(batch, masks=random_mask)
    random_masked_ce_loss = ce_vs_labels(random_masked_logits)
    random_masked_kl_loss = kl_vs_target(random_masked_logits)

    # we use rounded causal importances as masks
    rounded_ci = {k: (v > rounding_threshold).float() for k, v in ci.items()}
    rounded_masked_logits = model.forward_with_components(batch, masks=rounded_ci)
    rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
    rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

    # we zero all the components
    zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
    zero_masked_logits = model.forward_with_components(batch, masks=zero_masks)
    zero_masked_ce_loss = ce_vs_labels(zero_masked_logits)
    zero_masked_kl_loss = kl_vs_target(zero_masked_logits)

    target_model_ce_loss = ce_vs_labels(target_out)

    def pct_ce_unrecovered(ce: float) -> float:
        """pct of ce loss that is unrecovered, between zero masked and target model ce loss"""
        return (ce - target_model_ce_loss) / (zero_masked_ce_loss - target_model_ce_loss)

    def ce_difference(ce: float) -> float:
        """difference between ce loss and target model ce loss"""
        return ce - target_model_ce_loss

    return {
        # Not sure if these are useful:
        # "ce/target": target_model_ce_loss,
        # "ce/ci_masked": ci_masked_ce_loss,
        # "ce/unmasked": unmasked_ce_loss,
        # "ce/stoch_masked": stoch_masked_ce_loss,
        # "ce/random_masked": random_masked_ce_loss,
        # "ce/rounded_masked": rounded_masked_ce_loss,
        # "ce/zero_masked": zero_masked_ce_loss,
        "kl/ci_masked": ci_masked_kl_loss,
        "kl/unmasked": unmasked_kl_loss,
        "kl/stoch_masked": stoch_masked_kl_loss,
        "kl/random_masked": random_masked_kl_loss,
        "kl/rounded_masked": rounded_masked_kl_loss,
        "kl/zero_masked": zero_masked_kl_loss,
        "ce_difference/ci_masked": ce_difference(ci_masked_ce_loss),
        "ce_difference/unmasked": ce_difference(unmasked_ce_loss),
        "ce_difference/stoch_masked": ce_difference(stoch_masked_ce_loss),
        "ce_difference/random_masked": ce_difference(random_masked_ce_loss),
        "ce_difference/rounded_masked": ce_difference(rounded_masked_ce_loss),
        "ce_unrecovered/ci_masked": pct_ce_unrecovered(ci_masked_ce_loss),
        "ce_unrecovered/unmasked": pct_ce_unrecovered(unmasked_ce_loss),
        "ce_unrecovered/stoch_masked": pct_ce_unrecovered(stoch_masked_ce_loss),
        "ce_unrecovered/random_masked": pct_ce_unrecovered(random_masked_ce_loss),
        "ce_unrecovered/rounded_masked": pct_ce_unrecovered(rounded_masked_ce_loss),
        # no zero masked ce_unrecovered because it's tautologically 100%
    }


def lm_embed(config: Config, input_batches: list[MetricsBatchInputs]) -> Mapping[str, wandb.Table]:
    causal_importances: list[Float[Tensor, "... C"]] = []

    for input in input_batches:
        assert len(input.ci) == 1, "Only one embedding component allowed"
        causal_importances.append(next(iter(input.ci.values())))

    assert len(causal_importances) == 1, "Only one embedding component allowed"
    key = next(iter(causal_importances))

    assert key == "transformer.wte" or key == "model.embed_tokens"
    all_ci = torch.cat(causal_importances)

    N_SAMPLES = 20
    N_COMPONENTS = 10

    # Create a 20x10 table for wandb
    table_data = []
    # Add "Row Name" as the first column
    component_names = ["TokenSample"] + ["CompVal" for _ in range(N_COMPONENTS)]

    for i, ci in enumerate(all_ci[0, :N_SAMPLES]):
        active_values = ci[ci > config.ci_alive_threshold].tolist()
        # Cap at 10 components
        active_values = active_values[:N_COMPONENTS]
        formatted_values = [f"{val:.2f}" for val in active_values]
        # Pad with empty strings if fewer than 10 components
        while len(formatted_values) < N_COMPONENTS:
            formatted_values.append("0")
        # Add row name as the first element
        table_data.append([f"{i}"] + formatted_values)

    return {"embed_ci_sample": wandb.Table(data=table_data, columns=component_names)}


METRICS_FNS: dict[str, CreateMetricsFn] = {
    fn.__name__: fn
    for fn in [
        l0,
        ce_kl,
        lm_embed,
    ]
}


def create_metrics(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    device: str,
    config: Config,
) -> dict[str, float | int | wandb.Table]:
    """Create metrics for logging."""

    # get inputs
    inputs_list: list[MetricsBatchInputs] = []
    for _ in range(n_eval_steps):
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)
        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )
        ci, _ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=config.sigmoid_type
        )

        inputs_list.append(
            MetricsBatchInputs(
                batch=batch,
                target_out=target_out,
                ci=ci,
            )
        )

    metrics: dict[str, float | int | wandb.Table] = {}
    for fn_cfg in config.metrics_fns:
        if (fn := METRICS_FNS.get(fn_cfg.name)) is None:
            raise ValueError(f"Metric {fn_cfg.name} not found in METRICS_FNS")

        result = fn(config, inputs_list, **fn_cfg.extra_kwargs)

        if already_present_keys := set(result.keys()).intersection(metrics.keys()):
            raise ValueError(f"Metric keys {already_present_keys} already exists in metrics")

        metrics.update(result)

    return metrics
