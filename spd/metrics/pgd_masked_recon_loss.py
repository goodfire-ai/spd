from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce


def pgd_recon_loss(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        target_out=target_out,
        output_loss_type=output_loss_type,
        routing="all",  # <- Key difference from pgd_masked_recon_subset_loss.py
        pgd_config=pgd_config,
    )
    return sum_loss / n_examples


class PGDReconLoss(Metric):
    """Recon loss when masking with adversarially-optimized values and routing to all component
    layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        pgd_config: PGDConfig,
        use_delta_component: bool,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.use_delta_component: bool = use_delta_component
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = pgd_masked_recon_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            target_out=target_out,
            output_loss_type=self.output_loss_type,
            routing="all",
            pgd_config=self.pgd_config,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
