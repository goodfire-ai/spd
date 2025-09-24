from collections import defaultdict
from collections.abc import Mapping
from fnmatch import fnmatch
from typing import Any, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_kl_divergence_lm


class SubsetReconstructionLoss(Metric):
    """Compute reconstruction loss for specific subsets of components."""

    slow = False
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        include_patterns: dict[str, list[str]] | None = None,
        exclude_patterns: dict[str, list[str]] | None = None,
        n_mask_samples: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.n_mask_samples = n_mask_samples
        self.include_patterns = include_patterns or {}
        self.exclude_patterns = exclude_patterns or {}

        if not self.include_patterns and not self.exclude_patterns:
            raise ValueError(
                "At least one of include_patterns or exclude_patterns must be provided"
            )

        self.losses: dict[str, list[float]] = defaultdict(list)

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **kwargs: Any,
    ) -> None:
        losses = self._calc_subset_losses(
            batch=batch, target_out=target_out, ci=ci, weight_deltas=weight_deltas
        )
        for key, value in losses.items():
            self.losses[key].append(value)

    @override
    def compute(self) -> Mapping[str, float | str]:
        results = {k: sum(v) / len(v) for k, v in self.losses.items()}

        metrics_by_type = {"kl": {}, "ce": {}, "ce_unrec": {}}
        for key, value in results.items():
            if not key.startswith("subset/"):
                continue
            parts = key.split("/")
            if len(parts) != 3:
                continue
            subset_name = parts[1]
            metric_type = parts[2]
            if metric_type.endswith("_all_ones"):
                continue
            if metric_type == "kl":
                metrics_by_type["kl"][subset_name] = value
            elif metric_type == "ce":
                metrics_by_type["ce"][subset_name] = value
            elif metric_type == "ce_unrec":
                metrics_by_type["ce_unrec"][subset_name] = value

        for metric_type, subset_values in metrics_by_type.items():
            if subset_values:
                worst_subset = max(subset_values, key=lambda k: subset_values[k])
                worst_value = subset_values[worst_subset]
                results[f"subset_worst/{metric_type}"] = worst_value
                results[f"subset_worst/{metric_type}_subset"] = worst_subset

        return results

    def _calc_subset_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Tensor],
    ) -> Mapping[str, float]:
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
                sampling=self.config.sampling,
                routing="all",
                weight_deltas=weight_deltas if self.config.use_delta_component else None,
            )
            for _ in range(self.n_mask_samples)
        ]
        results = {}
        all_modules = list(ci.keys())

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

            results[f"subset/{name}/kl"] = mean_kl
            results[f"subset/{name}/ce"] = mean_ce
            results[f"subset/{name}/ce_unrec"] = ce_unrec

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

            results[f"subset/{name}/kl"] = mean_kl
            results[f"subset/{name}/ce"] = mean_ce
            results[f"subset/{name}/ce_unrec"] = ce_unrec

        return results
