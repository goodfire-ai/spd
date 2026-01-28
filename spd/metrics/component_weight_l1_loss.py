from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import all_reduce


def _component_weight_l1_loss_update(
    model: ComponentModel,
) -> tuple[Float[Tensor, ""], int]:
    """Compute sum of absolute values of V and U matrices across all components.

    Normalized by the number of parameters in V and U matrices.
    """
    device = next(model.parameters()).device
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0
    for components in model.components.values():
        sum_loss += components.V.abs().sum()
        sum_loss += components.U.abs().sum()
        total_params += components.V.numel() + components.U.numel()
    return sum_loss, total_params


def _component_weight_l1_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def component_weight_l1_loss(model: ComponentModel) -> Float[Tensor, ""]:
    """L1 regularization loss on component V and U matrices."""
    sum_loss, total_params = _component_weight_l1_loss_update(model)
    return _component_weight_l1_loss_compute(sum_loss, total_params)


class ComponentWeightL1Loss(Metric):
    """L1 regularization on component V and U matrices."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_params = torch.tensor(0, device=device)

    @override
    def update(self, **_: Any) -> None:
        sum_loss, total_params = _component_weight_l1_loss_update(self.model)
        self.sum_loss += sum_loss
        self.total_params += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        total_params = all_reduce(self.total_params, op=ReduceOp.SUM)
        return _component_weight_l1_loss_compute(sum_loss, total_params)
