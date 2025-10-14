from typing import Any, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.component_model import ComponentModel
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
        ci=ci,
        weight_deltas=weight_deltas,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        routing="all",  # <- Key difference from pgd_masked_recon_subset_loss.py
        pgd_config=pgd_config,
    )
    return sum_loss / n_examples


class PGDReconLoss(Metric):
    """Recon loss when masking with adversarially-optimized values and routing to all component layers."""

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
        self.sum_losses = [torch.tensor(0.0, device=device) for _ in range(pgd_config.n_steps)]
        self.n_exampleses = [torch.tensor(0, device=device) for _ in range(pgd_config.n_steps)]

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        for i in range(self.pgd_config.n_steps):
            sum_loss, n_examples = pgd_masked_recon_loss_update(
                model=self.model,
                ci=ci,
                weight_deltas=weight_deltas if self.use_delta_component else None,
                output_loss_type=self.output_loss_type,
                batch=batch,
                target_out=target_out,
                routing="all",
                pgd_config=PGDConfig(
                    init=self.pgd_config.init,
                    step_size=self.pgd_config.step_size,
                    n_steps=i + 1,
                    mask_scope=self.pgd_config.mask_scope,
                ),
            )
            self.sum_losses[i] += sum_loss
            self.n_exampleses[i] += n_examples

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        out: dict[str, Float[Tensor, ""]] = {}
        for i in range(self.pgd_config.n_steps):
            sum_loss = all_reduce(self.sum_losses[i], op=ReduceOp.SUM)
            n_examples = all_reduce(self.n_exampleses[i], op=ReduceOp.SUM)
            out[f"pgd_loss_{i}_steps"] = sum_loss / n_examples
        return out
        # sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        # n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        # return sum_loss / n_examples
