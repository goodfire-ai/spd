from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _ci_masked_recon_layerwise_loss_update(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl", "mem", "mem_ce"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    labels: Int[Tensor, "batch"] | None = None,  # noqa: F821
) -> tuple[Float[Tensor, ""], int]:
    sum_loss = torch.tensor(0.0, device=batch.device)
    n_examples = 0
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    for module_name, mask_info in mask_infos.items():
        out = model(batch, mask_infos={module_name: mask_info})
        loss = calc_sum_recon_loss_lm(
            pred=out, target=target_out, loss_type=output_loss_type, labels=labels
        )
        if output_loss_type == "mse":
            n_examples += out.shape.numel()
        elif output_loss_type in ("mem", "mem_ce"):
            n_examples += out.shape[0]  # batch size only for mem/mem_ce
        else:  # kl
            n_examples += out.shape[:-1].numel()
        sum_loss += loss
    return sum_loss, n_examples


def _ci_masked_recon_layerwise_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_layerwise_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl", "mem", "mem_ce"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    labels: Int[Tensor, "batch"] | None = None,  # noqa: F821
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_layerwise_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        labels=labels,
    )
    return _ci_masked_recon_layerwise_loss_compute(sum_loss, n_examples)


class CIMaskedReconLayerwiseLoss(Metric):
    """Recon loss when masking with CI values directly one layer at a time."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl", "mem", "mem_ce"],
    ) -> None:
        self.model = model
        self.output_loss_type: Literal["mse", "kl", "mem", "mem_ce"] = output_loss_type
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        labels: Int[Tensor, "batch"] | None = None,  # noqa: F821
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _ci_masked_recon_layerwise_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            labels=labels,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _ci_masked_recon_layerwise_loss_compute(sum_loss, n_examples)
