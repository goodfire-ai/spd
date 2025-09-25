from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _stochastic_recon_layerwise_loss_update(
    model: ComponentModel,
    config: Config,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    assert weight_deltas, "Empty weight deltas"
    device = next(iter(ci.values())).device
    sum_loss = torch.tensor(0.0, device=device)
    n_examples = 0

    stochastic_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            sampling=config.sampling,
            routing="all",
            weight_deltas=weight_deltas if config.use_delta_component else None,
        )
        for _ in range(config.n_mask_samples)
    ]

    for stochastic_mask_infos in stochastic_mask_infos_list:
        for module_name, mask_info in stochastic_mask_infos.items():
            out = model(batch, mode="components", mask_infos={module_name: mask_info})
            loss_type = config.output_loss_type
            loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)

            n_examples += out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
            sum_loss += loss
    return sum_loss, n_examples


def _stochastic_recon_layerwise_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def stochastic_recon_layerwise_loss(
    model: ComponentModel,
    config: Config,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _stochastic_recon_layerwise_loss_update(
        model, config, batch, target_out, ci, weight_deltas
    )
    return _stochastic_recon_layerwise_loss_compute(sum_loss, n_examples)


class StochasticReconLayerwiseLoss(Metric):
    """Recon loss when sampling with stochastic masks one layer at a time."""

    slow = False
    is_differentiable: bool | None = True
    full_state_update: bool | None = False  # Avoid double update calls

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
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **kwargs: Any,
    ) -> None:
        sum_loss, n_examples = _stochastic_recon_layerwise_loss_update(
            model=self.model,
            config=self.config,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        return _stochastic_recon_layerwise_loss_compute(self.sum_loss, self.n_examples)
