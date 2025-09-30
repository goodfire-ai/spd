from typing import Any, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _ci_masked_recon_loss_update(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
) -> tuple[Float[Tensor, ""], int]:
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    out = model(batch, mode="components", mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    return loss, out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()


def _ci_masked_recon_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
    )
    return _ci_masked_recon_loss_compute(sum_loss, n_examples)


class CIMaskedReconLoss(Metric):
    """Recon loss when masking with CI values directly on all component layers."""

    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_examples: Int[Tensor, ""]

    def __init__(
        self, model: ComponentModel, output_loss_type: Literal["mse", "kl"], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type

        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _ci_masked_recon_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        return _ci_masked_recon_loss_compute(self.sum_loss, self.n_examples)
