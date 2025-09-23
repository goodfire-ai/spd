from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.mask_info import make_mask_infos
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import calc_kl_divergence_lm


class CIReconLayerwiseLoss(Metric):
    """Calculate the recon loss when masking with CI values directly one layer at a time."""

    slow = False
    is_differentiable: bool | None = True

    sum_ci_recon_layerwise: Float[Tensor, ""]
    n_examples: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

        self.add_state("sum_ci_recon_layerwise", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
        for module_name, mask_info in mask_infos.items():
            out = self.model(batch, mode="components", mask_infos={module_name: mask_info})
            if self.config.output_loss_type == "mse":
                loss = ((out - target_out) ** 2).sum()
            else:
                loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
            self.n_examples += out.shape[:-1].numel()
            self.sum_ci_recon_layerwise += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_ci_recon_layerwise / self.n_examples
