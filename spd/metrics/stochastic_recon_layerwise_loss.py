from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.mask_info import WeightDeltaAndMask, make_mask_infos
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm


class StochasticReconLayerwiseLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_stochastic_layerwise_recon: Float[Tensor, ""]
    n_examples: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.add_state(
            "sum_stochastic_layerwise_recon", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **kwargs: Any,
    ) -> None:
        stoch_masks_list = calc_stochastic_masks(
            causal_importances=ci,
            n_mask_samples=self.config.n_mask_samples,
            sampling=self.config.sampling,
        )
        for stoch_masks in stoch_masks_list:
            deltas_and_masks: dict[str, WeightDeltaAndMask] | None = (
                {k: (weight_deltas[k], stoch_masks.weight_delta_masks[k]) for k in weight_deltas}
                if self.config.use_delta_component
                else None
            )
            mask_infos = make_mask_infos(
                masks=stoch_masks.component_masks,
                weight_deltas_and_masks=deltas_and_masks,
            )

            for module_name, mask_info in mask_infos.items():
                out = self.model(batch, mode="components", mask_infos={module_name: mask_info})
                if self.config.output_loss_type == "mse":
                    loss = ((out - target_out) ** 2).sum()
                else:
                    loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
                self.n_examples += (
                    out.shape.numel()
                    if self.config.output_loss_type == "mse"
                    else out.shape[:-1].numel()
                )
                self.sum_stochastic_layerwise_recon += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_stochastic_layerwise_recon / self.n_examples
