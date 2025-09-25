from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.general_utils import calc_recon_loss_lm


class CIMaskedReconLayerwiseLoss(Metric):
    """Recon loss when masking with CI values directly one layer at a time."""

    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_examples: Int[Tensor, ""]

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

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
        loss_type = self.config.output_loss_type

        mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
        for module_name, mask_info in mask_infos.items():
            # TODO: Refactor this accumulation, it's used in lots of losses
            out = self.model(batch, mode="components", mask_infos={module_name: mask_info})
            loss = calc_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
            self.n_examples += out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
            self.sum_loss += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_loss / self.n_examples
