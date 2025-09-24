from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.general_utils import calc_kl_divergence_lm


class CIMaskedReconLoss(Metric):
    """Recon loss when masking with CI values directly on all component layers."""

    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_examples: Int[Tensor, ""]

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.output_loss_type = config.output_loss_type

        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
        out = self.model(batch, mode="components", mask_infos=mask_infos)
        if self.output_loss_type == "mse":
            loss = ((out - target_out) ** 2).sum()
        else:
            loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
        self.n_examples += (
            out.shape.numel() if self.output_loss_type == "mse" else out.shape[:-1].numel()
        )
        self.sum_loss += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_loss / self.n_examples
