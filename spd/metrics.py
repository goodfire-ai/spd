"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, override

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel

# from spd.plotting import create_embed_ci_sample_table
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data


class Metric(ABC):
    @abstractmethod
    def watch_batch(self, batch: Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, float | int | wandb.Table]: ...


class L0(Metric):
    def __init__(self, model: ComponentModel, l0_threshold: float):
        self.model = model
        self.l0_threshold = l0_threshold
        self.l0s = defaultdict[str, list[float]](list)

    @override
    def watch_batch(self, batch: Tensor) -> None:
        _, pre_weight_acts = self.model.forward_with_pre_forward_cache_hooks(
            batch, module_names=self.model.target_module_paths
        )
        causal_importances, _ = self.model.calc_causal_importances(pre_weight_acts)
        for layer_name, ci in causal_importances.items():
            mean_dims = tuple(range(ci.ndim - 1))
            l0 = (ci > self.l0_threshold).float().mean(dim=mean_dims).sum().item()
            self.l0s[layer_name].append(l0)

    @override
    def compute(self) -> Mapping[str, float]:
        out = {}
        for layer_name, l0s in self.l0s.items():
            out[f"misc/l0/{layer_name}"] = sum(l0s) / len(l0s)
        return out


class KLDivergences(Metric):
    def __init__(self, model: ComponentModel):
        self.model = model
        self.kl_divergences = defaultdict[str, list[float]](list)

    @override
    def watch_batch(self, batch: Tensor) -> None:
        target_out, pre_weight_acts = self.model.forward_with_pre_forward_cache_hooks(
            batch, module_names=self.model.target_module_paths
        )

        causal_importances, _ = self.model.calc_causal_importances(pre_weight_acts)
        masked_component_out = self.model.forward_with_components(batch, masks=causal_importances)

        nonmask = {k: torch.ones_like(v) for k, v in causal_importances.items()}
        unmasked_component_out = self.model.forward_with_components(batch, masks=nonmask)

        kl_vs_target = calc_kl_divergence_lm(pred=unmasked_component_out, target=target_out)
        kl_vs_target_masked = calc_kl_divergence_lm(pred=masked_component_out, target=target_out)

        self.kl_divergences["unmasked_kl_loss_vs_target"].append(kl_vs_target.item())
        self.kl_divergences["masked_kl_loss_vs_target"].append(kl_vs_target_masked.item())

    @override
    def compute(self) -> Mapping[str, float]:
        return {k: sum(v) / len(v) for k, v in self.kl_divergences.items()}


def _calc_ce_and_kl_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    rounding_threshold: float,
) -> Mapping[str, float]:
    assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

    # make sure labels don't "wrap around": you **can't** predict the first token.
    masked_batch = batch.clone()
    masked_batch[:, 0] = -100  # F.cross_entropy ignores -99
    flat_masked_batch = masked_batch.flatten()

    def ce_vs_labels(logits: Tensor) -> float:
        flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
        return F.cross_entropy(flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100).item()

    target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=model.target_module_paths
    )

    def kl_vs_target(logits: Tensor) -> float:
        return calc_kl_divergence_lm(pred=logits, target=target_out).item()

    # CE When...
    # we use the causal importances as a mask
    ci, _ci_upper_leaky = model.calc_causal_importances(pre_weight_acts)
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


class CEandKLLosses(Metric):
    def __init__(self, model: ComponentModel, rounding_threshold: float):
        self.model = model
        self.rounding_threshold = rounding_threshold
        self.ce_losses = defaultdict[str, list[float]](list)

    @override
    def watch_batch(self, batch: Int[Tensor, "..."]) -> None:
        ce_losses = _calc_ce_and_kl_losses(self.model, batch, self.rounding_threshold)
        for key, value in ce_losses.items():
            self.ce_losses[key].append(value)

    @override
    def compute(self) -> Mapping[str, float]:
        # I think this is fine: avg(sum(x)) = sum(avg(x))
        return {k: sum(v) / len(v) for k, v in self.ce_losses.items()}


class LMEmbedSampleTable(Metric):
    def __init__(self, model: ComponentModel, ci_alive_threshold: float):
        self.model = model
        self.ci_alive_threshold = ci_alive_threshold
        self.causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def watch_batch(self, batch: Tensor) -> None:
        _, pre_weight_acts = self.model.forward_with_pre_forward_cache_hooks(
            batch, module_names=self.model.target_module_paths
        )
        causal_importances, _ = self.model.calc_causal_importances(pre_weight_acts)
        for key, value in causal_importances.items():
            self.causal_importances[key].append(value)

    def _create_embed_ci_sample_table(
        self, causal_importances: Float[Tensor, "... C"]
    ) -> wandb.Table:
        """Create a wandb table visualizing embedding mask values.

        Args:
            causal_importances: Dictionary of causal importances for each component.

        Returns:
            A wandb Table object.
        """
        N_SAMPLES = 20

        # Create a 20x10 table for wandb
        table_data = []
        # Add "Row Name" as the first column
        component_names = ["TokenSample"] + ["CompVal" for _ in range(10)]

        for i, ci in enumerate(causal_importances[0, :N_SAMPLES]):
            active_values = ci[ci > self.ci_alive_threshold].tolist()
            # Cap at 10 components
            active_values = active_values[:10]
            formatted_values = [f"{val:.2f}" for val in active_values]
            # Pad with empty strings if fewer than 10 components
            while len(formatted_values) < 10:
                formatted_values.append("0")
            # Add row name as the first element
            table_data.append([f"{i}"] + formatted_values)

        return wandb.Table(data=table_data, columns=component_names)

    @override
    def compute(self) -> Mapping[str, wandb.Table]:
        assert len(self.causal_importances) == 1, "Only one embedding component allowed"
        key = next(iter(self.causal_importances))
        assert key == "transformer.wte" or key == "model.embed_tokens"
        ci: Float[Tensor, "... C"] = torch.cat(self.causal_importances[key])
        return {"misc/embed_ci_sample": self._create_embed_ci_sample_table(ci)}


def create_metrics(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    device: str,
    config: Config,
) -> Mapping[str, float | int | wandb.Table]:
    """Create metrics for logging."""
    out: dict[str, float | int | wandb.Table] = {}

    l0_metric = L0(model, config.ci_alive_threshold)
    ce_kl_metric = CEandKLLosses(model, config.ci_alive_threshold)
    embed_ci_metric = LMEmbedSampleTable(model, config.ci_alive_threshold)

    for _ in range(n_eval_steps):
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)

        l0_metric.watch_batch(batch)
        ce_kl_metric.watch_batch(batch)
        embed_ci_metric.watch_batch(batch)

    out.update(l0_metric.compute())
    out.update(ce_kl_metric.compute())
    out.update(embed_ci_metric.compute())

    return out

#     # for fn_cfg in config.metrics_fns:
#     #     if (fn := METRICS_FNS.get(fn_cfg.name)) is None:
#     #         raise ValueError(f"Metric {fn_cfg.name} not found in METRICS_FNS")

#     #     result = fn(inputs, **fn_cfg.extra_kwargs)

#     #     if already_present_keys := set(result.keys()).intersection(metrics.keys()):
#     #         raise ValueError(f"Metric keys {already_present_keys} already exists in metrics")

#     #     metrics.update(result)

#     return metrics


# # METRICS: dict[str, 
# #     fn.__name__: fn
# #     for fn in [

# #         # ci_l0,
# #         # lm_kl,
# #         # lm_embed,
# #         # lm_ce_losses,
# #     ]
# # }
