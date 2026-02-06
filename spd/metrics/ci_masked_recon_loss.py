from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.metrics.ci_masked_recon_subset_loss import ci_masked_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import all_reduce


def ci_masked_recon_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
) -> Float[Tensor, ""]:
    sum_loss, n_examples = ci_masked_recon_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        router=AllLayersRouter(),
    )
    return sum_loss / n_examples


class CIMaskedReconLoss(Metric):
    """Recon loss when masking with CI values directly on all component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self, model: ComponentModel, device: str, output_loss_type: Literal["mse", "kl"]
    ) -> None:
        self.model = model
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = ci_masked_recon_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            router=AllLayersRouter(),
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
