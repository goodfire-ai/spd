from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import OutputLossType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_n_recon_examples_lm, calc_sum_recon_loss_lm


def _ci_masked_recon_layerwise_loss_update(
    model: ComponentModel,
    output_loss_type: OutputLossType,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
) -> tuple[Float[Tensor, ""], int]:
    sum_loss = torch.tensor(0.0, device=batch.device)
    n_examples = 0
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    for module_name, mask_info in mask_infos.items():
        out = model(batch, mask_infos={module_name: mask_info})
        loss = calc_sum_recon_loss_lm(
            pred=out, target=target_out, loss_type=output_loss_type, batch=batch
        )
        n_examples += calc_n_recon_examples_lm(out.shape, output_loss_type)
        sum_loss += loss
    return sum_loss, n_examples


def _ci_masked_recon_layerwise_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_layerwise_loss(
    model: ComponentModel,
    output_loss_type: OutputLossType,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_layerwise_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
    )
    return _ci_masked_recon_layerwise_loss_compute(sum_loss, n_examples)


class CIMaskedReconLayerwiseLoss(Metric):
    """Recon loss when masking with CI values directly one layer at a time."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self, model: ComponentModel, device: str, output_loss_type: OutputLossType
    ) -> None:
        self.model = model
        self.output_loss_type: OutputLossType = output_loss_type
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
        sum_loss, n_examples = _ci_masked_recon_layerwise_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _ci_masked_recon_layerwise_loss_compute(sum_loss, n_examples)
