from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_kl_divergence_lm


class StochasticReconLoss(Metric):
    """Recon loss when sampling with stochastic masks on all component layers."""

    slow = False
    is_differentiable: bool | None = True

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
        stoch_mask_infos_list = [
            calc_stochastic_component_mask_info(
                causal_importances=ci,
                sampling=self.config.sampling,
                routing="all",
                weight_deltas=weight_deltas if self.config.use_delta_component else None,
            )
            for _ in range(self.config.n_mask_samples)
        ]
        for stoch_mask_infos in stoch_mask_infos_list:
            out = self.model(batch, mode="components", mask_infos=stoch_mask_infos)
            match self.config.output_loss_type:
                case "mse":
                    loss = ((out - target_out) ** 2).sum()
                case "kl":
                    loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
            self.n_examples += (
                out.shape.numel()
                if self.config.output_loss_type == "mse"
                else out.shape[:-1].numel()
            )
            self.sum_loss += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_loss / self.n_examples
