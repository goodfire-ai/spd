from typing import Any, Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Int
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_kl_divergence_lm


class CEandKLLosses(Metric):
    """CE and KL losses for different masking strategies.

    NOTE: Assumes all batches and sequences are the same size.
    """

    is_differentiable: bool | None = False

    # NOTE: Gross that we have to hardcode these here. Open to other ideas.
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

    n_positions: Int[Tensor, ""]  # batch_size * seq_len * n_batches_seen

    def __init__(
        self,
        model: ComponentModel,
        sampling: Literal["continuous", "binomial"],
        rounding_threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.sampling: Literal["continuous", "binomial"] = sampling
        self.rounding_threshold = rounding_threshold

        for key in self.loss_keys:
            self.add_state(
                f"{key}_sum",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        self.add_state("n_positions", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        *,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        **_: Any,
    ) -> None:
        ce_losses = self._calc_ce_and_kl_losses(batch=batch, target_out=target_out, ci=ci)

        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
        n_positions_in_batch = batch.shape[0] * batch.shape[1]

        for key in self.loss_keys:
            key_sum = getattr(self, f"{key}_sum")
            key_sum += ce_losses[key] * n_positions_in_batch
        self.n_positions += n_positions_in_batch

    @override
    def compute(self) -> dict[str, float]:
        losses = {}
        for key in self.loss_keys:
            loss_sum = getattr(self, f"{key}_sum")
            losses[key] = loss_sum / self.n_positions
        return losses

    def _calc_ce_and_kl_losses(
        self, batch: Tensor, target_out: Tensor, ci: dict[str, Tensor]
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

        ci_mask_infos = make_mask_infos(ci)
        ci_masked_logits = self.model(batch, mode="components", mask_infos=ci_mask_infos)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        # Sample stochastic masks based on the causal importances
        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            sampling=self.sampling,
            routing="all",
            weight_deltas=None,
        )
        stoch_masked_logits = self.model(batch, mode="components", mask_infos=mask_infos)
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        nonmask_infos = make_mask_infos({k: torch.ones_like(v) for k, v in ci.items()})
        unmasked_logits = self.model(batch, mode="components", mask_infos=nonmask_infos)
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        # Completely random masks
        rand_mask_infos = make_mask_infos({k: torch.rand_like(v) for k, v in ci.items()})
        random_masked_logits = self.model(batch, mode="components", mask_infos=rand_mask_infos)
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        # Rounded causal importances as masks
        rounded_mask_infos = make_mask_infos(
            {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        )
        rounded_masked_logits = self.model(batch, mode="components", mask_infos=rounded_mask_infos)
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        # Zero all the components
        zero_mask_infos = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.items()})
        zero_masked_logits = self.model(batch, mode="components", mask_infos=zero_mask_infos)
        zero_masked_ce_loss = ce_vs_labels(zero_masked_logits)
        zero_masked_kl_loss = kl_vs_target(zero_masked_logits)

        target_model_ce_loss = ce_vs_labels(target_out)

        def pct_ce_unrecovered(ce: float) -> float:
            return (ce - target_model_ce_loss) / (zero_masked_ce_loss - target_model_ce_loss)

        def ce_difference(ce: float) -> float:
            return ce - target_model_ce_loss

        out: dict[str, float] = {
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
        assert list(out.keys()) == self.loss_keys
        return out
