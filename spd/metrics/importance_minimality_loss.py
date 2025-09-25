from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric


def _importance_minimality_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float, eps: float
) -> tuple[Float[Tensor, " C"], int]:
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    device = next(iter(ci_upper_leaky.values())).device
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0
    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # Note: layer_ci_upper_leaky already >= 0
        sum_loss += ((layer_ci_upper_leaky + eps) ** pnorm).sum()
        total_params += layer_ci_upper_leaky.shape[:-1].numel()
    return sum_loss, total_params


def _importance_minimality_loss_compute(
    sum_loss: Float[Tensor, " C"], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float, eps: float
) -> Float[Tensor, ""]:
    sum_loss, total_params = _importance_minimality_loss_update(ci_upper_leaky, pnorm, eps)
    return _importance_minimality_loss_compute(sum_loss, total_params)


class ImportanceMinimalityLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, " C"]
    n_examples: Int[Tensor, ""]

    def __init__(self, *args: Any, pnorm: Any, eps: Any = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pnorm = float(pnorm)
        self.eps = float(eps)
        self.add_state("sum_loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, *, ci_upper_leaky: dict[str, Float[Tensor, "... C"]], **kwargs: Any) -> None:
        sum_loss, total_params = _importance_minimality_loss_update(
            ci_upper_leaky, self.pnorm, self.eps
        )
        self.sum_loss += sum_loss
        self.n_examples += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        return _importance_minimality_loss_compute(self.sum_loss, self.n_examples)
