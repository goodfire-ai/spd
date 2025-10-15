from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import CoeffSchedule
from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.scheduling import get_value
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _importance_minimality_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float | CoeffSchedule,
    eps: float,
    current_frac_of_training: float,
) -> tuple[Float[Tensor, " C"], int]:
    """Calculate the sum of the importance minimality loss over all layers.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.
    """
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    pnorm = get_value(value=pnorm, current_frac_of_training=current_frac_of_training)
    device = get_obj_device(ci_upper_leaky)
    sum_loss = torch.tensor(0.0, device=device)
    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # Note: layer_ci_upper_leaky already >= 0
        sum_loss += ((layer_ci_upper_leaky + eps) ** pnorm).sum()
    n_params = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return sum_loss, n_params


def _importance_minimality_loss_compute(
    sum_loss: Float[Tensor, " C"], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    current_frac_of_training: float,
    eps: float,
    pnorm: float | CoeffSchedule,
) -> Float[Tensor, ""]:
    sum_loss, total_params = _importance_minimality_loss_update(
        ci_upper_leaky=ci_upper_leaky,
        pnorm=pnorm,
        eps=eps,
        current_frac_of_training=current_frac_of_training,
    )
    return _importance_minimality_loss_compute(sum_loss, total_params)


class ImportanceMinimalityLoss(Metric):
    """L_p loss on the sum of CI values.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.

    Args:
        pnorm: The p value for the L_p norm or a schedule for it
        eps: The epsilon value for numerical stability.
    """

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pnorm: float | CoeffSchedule,
        eps: float = 1e-12,
    ) -> None:
        self.pnorm = pnorm
        self.eps = eps
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        current_frac_of_training: float,
        **_: Any,
    ) -> None:
        sum_loss, total_params = _importance_minimality_loss_update(
            ci_upper_leaky=ci_upper_leaky,
            pnorm=self.pnorm,
            eps=self.eps,
            current_frac_of_training=current_frac_of_training,
        )
        self.sum_loss += sum_loss
        self.n_examples += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _importance_minimality_loss_compute(sum_loss, n_examples)
