from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

import einops
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.mask_info import make_mask_infos
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm


class CEandKLLosses(Metric):
    slow = False
    is_differentiable: bool | None = False

    # TODO: Awful that we have to hardcode these here and hope they're the same as the output
    # of _calc_ce_and_kl_losses
    loss_keys: list[str] = [
        "kl/ci_masked",
        "kl/unmasked",
        "kl/stoch_masked",
        "kl/random_masked",
        "kl/rounded_masked",
        "kl/zero_masked",
        "ce_difference/ci_masked",
        "ce_difference/unmasked",
        "ce_difference/stoch_masked",
        "ce_difference/random_masked",
        "ce_difference/rounded_masked",
        "ce_unrecovered/ci_masked",
        "ce_unrecovered/unmasked",
        "ce_unrecovered/stoch_masked",
        "ce_unrecovered/random_masked",
        "ce_unrecovered/rounded_masked",
    ]

    def __init__(
        self, model: ComponentModel, config: Config, rounding_threshold: float, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.rounding_threshold = rounding_threshold
        for key in self.loss_keys:
            self.add_state(
                key,
                default=[],
                dist_reduce_fx="cat",
            )

    @override
    def update(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        **kwargs: Any,
    ) -> None:
        ce_losses = self._calc_ce_and_kl_losses(batch=batch, target_out=target_out, ci=ci)
        for key in self.loss_keys:
            getattr(self, key).append(ce_losses[key])

    @override
    def compute(self) -> Mapping[str, float]:
        losses = {}
        # TODO: I think this is very inefficient as it will gather across ranks for each key
        for key in self.loss_keys:
            loss_values: list[float] = getattr(self, key)
            losses[key] = sum(loss_values) / len(loss_values)
        return losses

    def _calc_ce_and_kl_losses(
        self, batch: Tensor, target_out: Tensor, ci: dict[str, Tensor]
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

        ci_mask_infos = make_mask_infos(ci)
        ci_masked_logits = self.model(batch, mode="components", mask_infos=ci_mask_infos)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        stoch_masks = [
            m.component_masks
            for m in calc_stochastic_masks(ci, n_mask_samples=1, sampling=self.config.sampling)
        ][0]
        stoch_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(stoch_masks)
        )
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        nonmask = {k: torch.ones_like(v) for k, v in ci.items()}
        unmasked_logits = self.model(batch, mode="components", mask_infos=make_mask_infos(nonmask))
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        rand_masks = {layer: torch.rand_like(v) for layer, v in ci.items()}
        random_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(rand_masks)
        )
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        rounded_ci = {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        rounded_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(rounded_ci)
        )
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
        zero_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(zero_masks)
        )
        zero_masked_ce_loss = ce_vs_labels(zero_masked_logits)
        zero_masked_kl_loss = kl_vs_target(zero_masked_logits)

        target_model_ce_loss = ce_vs_labels(target_out)

        def pct_ce_unrecovered(ce: float) -> float:
            return (ce - target_model_ce_loss) / (zero_masked_ce_loss - target_model_ce_loss)

        def ce_difference(ce: float) -> float:
            return ce - target_model_ce_loss

        return {
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
        }
