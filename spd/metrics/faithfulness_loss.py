from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel


class FaithfulnessLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_faithfulness: Float[Tensor, ""]
    total_params: int

    def __init__(self, model: ComponentModel, _config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.add_state("sum_faithfulness", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_params", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **kwargs: Any,
    ) -> None:
        for delta in weight_deltas.values():
            self.sum_faithfulness += (delta**2).sum()
            self.total_params += delta.numel()

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_faithfulness / self.total_params
