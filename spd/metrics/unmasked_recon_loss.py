from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _unmasked_recon_loss_update(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    all_ones_mask_infos = make_mask_infos(
        # (C,) will broadcast to (B, S, C)
        {
            module_path: torch.ones(model.module_to_c[module_path], device=batch.device)
            for module_path in model.target_module_paths
        }
    )
    out = model(batch, mask_infos=all_ones_mask_infos)
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
    return loss, n_examples


def _unmasked_recon_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def unmasked_recon_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _unmasked_recon_loss_update(
        model,
        output_loss_type,
        batch,
        target_out,
    )
    return _unmasked_recon_loss_compute(sum_loss, n_examples)


class UnmaskedReconLoss(Metric):
    """Recon loss using the unmasked components and without the delta component."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
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
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _unmasked_recon_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _unmasked_recon_loss_compute(sum_loss, n_examples)
