from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel


def _faithfulness_loss_update(
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
) -> tuple[Float[Tensor, ""], int]:
    assert weight_deltas, "Empty weight deltas"
    device = next(iter(weight_deltas.values())).device
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0
    for delta in weight_deltas.values():
        sum_loss += (delta**2).sum()
        total_params += delta.numel()
    return sum_loss, total_params


def _faithfulness_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def faithfulness_loss(weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]) -> Float[Tensor, ""]:
    sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
    return _faithfulness_loss_compute(sum_loss, total_params)


class FaithfulnessLoss(Metric):
    """MSE between the target weights and the sum of the components."""

    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    total_params: Int[Tensor, ""]

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_params", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **kwargs: Any,
    ) -> None:
        sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
        self.sum_loss += sum_loss
        self.total_params += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        return _faithfulness_loss_compute(self.sum_loss, self.total_params)
