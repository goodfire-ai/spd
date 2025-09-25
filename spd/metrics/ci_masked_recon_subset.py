from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import sample_uniform_k_subset_routing_masks
from spd.utils.general_utils import calc_sum_recon_loss_lm


class CIMaskedReconSubset(Metric):
    """Recon loss when masking with raw CI values and routing to subsets of component layers."""

    slow = False
    is_differentiable: bool | None = True
    full_state_update: bool | None = False  # Avoid double update calls

    sum_loss: Float[Tensor, ""]
    n_examples: Int[Tensor, ""]

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.add_state("sum_loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **kwargs: dict[str, Any],
    ) -> None:
        subset_routing_masks = sample_uniform_k_subset_routing_masks(
            mask_shape=next(iter(ci.values())).shape[:-1],
            module_names=list(ci.keys()),
            device=batch.device,
        )
        mask_infos = make_mask_infos(
            component_masks=ci,
            routing_masks=subset_routing_masks,
            weight_deltas_and_masks=None,
        )
        out = self.model(batch, mode="components", mask_infos=mask_infos)
        loss_type = self.config.output_loss_type
        loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
        self.n_examples += out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
        self.sum_loss += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_loss / self.n_examples
