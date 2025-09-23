from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric


class ImportanceMinimalityLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_imp_min: Float[Tensor, " C"]
    n_examples: int

    def __init__(self, *args: Any, pnorm: Any, eps: Any = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pnorm = float(pnorm)
        self.eps = float(eps)
        self.add_state("sum_imp_min", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, ci_upper_leaky: dict[str, Float[Tensor, "... C"]], **kwargs: Any) -> None:
        for layer_ci_upper_leaky in ci_upper_leaky.values():
            # Note: layer_ci_upper_leaky already >= 0
            self.sum_imp_min += ((layer_ci_upper_leaky + self.eps) ** self.pnorm).sum()
            self.n_examples += layer_ci_upper_leaky.shape[:-1].numel()

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_imp_min / self.n_examples
