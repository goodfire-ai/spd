from fnmatch import fnmatch
from typing import Any, Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_kl_divergence_lm


class StochasticReconSubsetCEAndKL(Metric):
    """Compute reconstruction loss for specific subsets of components.

    NOTE: Assumes all batches and sequences are the same size.
    """

    slow = False
    is_differentiable: bool | None = False
    full_state_update: bool | None = False  # Avoid double update calls

    def __init__(
        self,
        model: ComponentModel,
        sampling: Literal["continuous", "binomial"],
        use_delta_component: bool,
        n_mask_samples: int,
        include_patterns: dict[str, list[str]] | None = None,
        exclude_patterns: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.sampling: Literal["continuous", "binomial"] = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples = n_mask_samples
        self.include_patterns = include_patterns or {}
        self.exclude_patterns = exclude_patterns or {}

        if not self.include_patterns and not self.exclude_patterns:
            raise ValueError(
                "At least one of include_patterns or exclude_patterns must be provided"
            )

        # Avoid using e.g. "layers.*.mlp_in" as an attribute
        self.key_to_sanitized: dict[str, str] = {}
        self.sanitized_to_key: dict[str, str] = {}

        all_keys: list[str] = []
        if self.include_patterns:
            all_keys += list(self.include_patterns.keys())
        if self.exclude_patterns:
            all_keys += list(self.exclude_patterns.keys())

        for key in all_keys:
            sanitized_key_raw = key.replace(".", "-").replace("*", "all")
            for suffix in ["_kl", "_ce", "_ce_unrec"]:
                key = f"{sanitized_key_raw}{suffix}"
                sanitized_key = f"{sanitized_key_raw}{suffix}"
                self.key_to_sanitized[key] = sanitized_key
                self.sanitized_to_key[sanitized_key] = key
                self.add_state(sanitized_key, default=[], dist_reduce_fx="cat")

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **_: Any,
    ) -> None:
        losses = self._calc_subset_losses(
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            sampling=self.sampling,
            use_delta_component=self.use_delta_component,
            n_mask_samples=self.n_mask_samples,
        )
        for key, value in losses.items():
            sanitized_key = self.key_to_sanitized[key]
            getattr(self, sanitized_key).append(value)

    @override
    def compute(self) -> dict[str, float | str]:
        results: dict[str, float | str] = {}
        for sanitized_key, key in self.sanitized_to_key.items():
            vals: list[float] = getattr(self, sanitized_key)
            mean_val = sum(vals) / len(vals)  # Assume all batches are the same size
            results[key] = mean_val

        # Get the worst subset for each metric type
        for metric_type in ["kl", "ce", "ce_unrec"]:
            results_by_type = {k: v for k, v in results.items() if k.endswith(metric_type)}
            worst_subset = max(results_by_type, key=lambda k: results_by_type[k])
            worst_value = results_by_type[worst_subset]
            results[f"subset_worst/{metric_type}"] = worst_value
            results[f"subset_worst/{metric_type}_subset"] = worst_subset

        return results

    def _calc_subset_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Tensor],
        sampling: Literal["continuous", "binomial"],
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> dict[str, float]:
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

        masked_batch = batch.clone()
        masked_batch[:, 0] = -100
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        # Compute baselines for CE unrecovered
        target_ce = ce_vs_labels(target_out)

        zero_mask_infos = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.items()})
        zero_out = self.model(batch, mode="components", mask_infos=zero_mask_infos)
        zero_ce = ce_vs_labels(zero_out)

        # Generate stochastic masks
        masks_list: list[dict[str, ComponentsMaskInfo]] = [
            calc_stochastic_component_mask_info(
                ci,
                sampling=sampling,
                routing="all",
                weight_deltas=weight_deltas if use_delta_component else None,
            )
            for _ in range(n_mask_samples)
        ]
        results = {}
        all_modules = list(ci.keys())

        # TODO: Reduce duplication
        # Process include patterns
        for name, patterns in self.include_patterns.items():
            active = [m for m in all_modules if any(fnmatch(m, p) for p in patterns)]

            outputs: list[Float[Tensor, "... vocab"]] = []  # pyright: ignore[reportRedeclaration]
            for layers_masks in masks_list:
                inc_mask_infos: dict[str, ComponentsMaskInfo] = {m: layers_masks[m] for m in active}
                outputs.append(self.model(batch, mode="components", mask_infos=inc_mask_infos))

            kl_losses = [kl_vs_target(out) for out in outputs]
            ce_losses = [ce_vs_labels(out) for out in outputs]

            mean_kl = sum(kl_losses) / len(kl_losses)
            mean_ce = sum(ce_losses) / len(ce_losses)
            ce_unrec = (mean_ce - target_ce) / (zero_ce - target_ce) if zero_ce != target_ce else 0

            results[f"{name}_kl"] = mean_kl
            results[f"{name}_ce"] = mean_ce
            results[f"{name}_ce_unrec"] = ce_unrec

        # Process exclude patterns
        for name, exclude_patterns in self.exclude_patterns.items():
            active = [m for m in all_modules if not any(fnmatch(m, p) for p in exclude_patterns)]

            outputs: list[Float[Tensor, "... vocab"]] = []
            for layers_masks in masks_list:
                exc_mask_infos: dict[str, ComponentsMaskInfo] = {m: layers_masks[m] for m in active}
                outputs.append(self.model(batch, mode="components", mask_infos=exc_mask_infos))

            kl_losses = [kl_vs_target(out) for out in outputs]
            ce_losses = [ce_vs_labels(out) for out in outputs]

            mean_kl = sum(kl_losses) / len(kl_losses)
            mean_ce = sum(ce_losses) / len(ce_losses)
            ce_unrec = (mean_ce - target_ce) / (zero_ce - target_ce) if zero_ce != target_ce else 0

            results[f"{name}_kl"] = mean_kl
            results[f"{name}_ce"] = mean_ce
            results[f"{name}_ce_unrec"] = ce_unrec

        return results
