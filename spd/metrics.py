"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, override

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


class StreamingMetricCreator(ABC):
    @abstractmethod
    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any): ...

    @abstractmethod
    def watch_batch(
        self,
        batch: Tensor,
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, float | int | wandb.Table]: ...


class CI_L0(StreamingMetricCreator):
    def __init__(self, model: ComponentModel, l0_threshold: float):
        self.model = model
        self.l0_threshold = l0_threshold
        self.l0s = defaultdict[str, list[float]](list)

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        for layer_name, layer_ci in ci.items():
            l0 = (layer_ci > self.l0_threshold).float().sum(-1).mean().item()
            self.l0s[layer_name].append(l0)

    @override
    def compute(self) -> Mapping[str, float]:
        out = {}
        for layer_name, l0s in self.l0s.items():
            out[f"l0/{layer_name}"] = sum(l0s) / len(l0s)
        return out


class CEandKLLosses(StreamingMetricCreator):
    def __init__(self, model: ComponentModel, rounding_threshold: float):
        self.model = model
        self.rounding_threshold = rounding_threshold
        self.ce_losses = defaultdict[str, list[float]](list)

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        ce_losses = self._calc_ce_and_kl_losses(batch, target_out, ci)
        for key, value in ce_losses.items():
            self.ce_losses[key].append(value)

    def _calc_ce_and_kl_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> Mapping[str, float]:
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

        # make sure labels don't "wrap around": you **can't** predict the first token.
        masked_batch = batch.clone()
        masked_batch[:, 0] = -100  # F.cross_entropy ignores -99
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        # CE When...
        # we use the causal importances as a mask
        ci_masked_logits = self.model.forward_with_components(batch, masks=ci)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        # we use the regular stochastic masks
        stoch_masks = calc_stochastic_masks(ci, n_mask_samples=1)[0]
        stoch_masked_logits = self.model.forward_with_components(batch, masks=stoch_masks)
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        # we use all components
        nonmask = {k: torch.ones_like(v) for k, v in ci.items()}
        unmasked_logits = self.model.forward_with_components(batch, masks=nonmask)
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        # we use completely random masks
        random_mask = {k: torch.rand_like(v) for k, v in ci.items()}
        random_masked_logits = self.model.forward_with_components(batch, masks=random_mask)
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        # we use rounded causal importances as masks
        rounded_ci = {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        rounded_masked_logits = self.model.forward_with_components(batch, masks=rounded_ci)
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        # we zero all the components
        zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
        zero_masked_logits = self.model.forward_with_components(batch, masks=zero_masks)
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

    @override
    def compute(self) -> Mapping[str, float]:
        # I think this is fine: avg(sum(x)) = sum(avg(x))
        return {k: sum(v) / len(v) for k, v in self.ce_losses.items()}


class LMEmbedSampleTable(StreamingMetricCreator):
    def __init__(self, model: ComponentModel, ci_alive_threshold: float):
        self.model = model
        self.ci_alive_threshold = ci_alive_threshold
        self.causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        for layer_name, layer_ci in ci.items():
            self.causal_importances[layer_name].append(layer_ci)

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
        all_ci = torch.cat(self.causal_importances[key])

        return {"embed_ci_sample": self._create_embed_ci_sample_table(all_ci)}


METRIC_CLASSES = {cls.__name__: cls for cls in StreamingMetricCreator.__subclasses__()}


def create_metrics(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    device: str,
    config: Config,
) -> dict[str, float | int | wandb.Table]:
    """Create metrics for logging."""
    metrics: list[StreamingMetricCreator] = []
    for metric_config in config.metrics:
        metric_cls = METRIC_CLASSES[metric_config.classname]
        metrics.append(metric_cls(model, config, **metric_config.extra_init_kwargs))

    for _ in range(n_eval_steps):
        # Do the work that's common between metrics
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)
        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )
        ci, _ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=config.sigmoid_type
        )

        for metric in metrics:
            metric.watch_batch(batch, target_out, ci)

    out: dict[str, float | int | wandb.Table] = {}
    all_dicts = [metric.compute() for metric in metrics]
    for d in all_dicts:
        if set(d.keys()).intersection(out.keys()):
            raise ValueError(
                f"Keys {set(d.keys()).intersection(out.keys())} already in output, cannot merge"
            )
        out.update(d)

    return out
