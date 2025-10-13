from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _get_linear_annealed_p(
    current_frac_of_training: float,
    initial_p: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> float:
    """Calculate the linearly annealed p value for L_p sparsity loss.

    Args:
        current_frac_of_training: Current fraction of training
        initial_p: Starting p value
        p_anneal_start_frac: Fraction of training after which to start annealing
        p_anneal_final_p: Final p value to anneal to
        p_anneal_end_frac: Fraction of training when annealing ends. We stay at the final p value from this point onward

    Returns:
        Current p value based on linear annealing schedule
    """
    if p_anneal_final_p is None or p_anneal_start_frac >= 1.0:
        return initial_p

    assert p_anneal_end_frac >= p_anneal_start_frac, (
        f"p_anneal_end_frac ({p_anneal_end_frac}) must be >= "
        f"p_anneal_start_frac ({p_anneal_start_frac})"
    )

    if current_frac_of_training < p_anneal_start_frac:
        return initial_p
    elif current_frac_of_training >= p_anneal_end_frac:
        return p_anneal_final_p
    else:
        # Linear interpolation between start and end fractions
        progress = (current_frac_of_training - p_anneal_start_frac) / (
            p_anneal_end_frac - p_anneal_start_frac
        )
        return initial_p + (p_anneal_final_p - initial_p) * progress


def _importance_minimality_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    eps: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
    current_frac_of_training: float,
) -> tuple[Float[Tensor, " C"], int]:
    """Calculate the sum of the importance minimality loss over all layers.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.
    """
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    pnorm = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
    )
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
    pnorm: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> Float[Tensor, ""]:
    sum_loss, total_params = _importance_minimality_loss_update(
        ci_upper_leaky=ci_upper_leaky,
        pnorm=pnorm,
        eps=eps,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
        current_frac_of_training=current_frac_of_training,
    )
    return _importance_minimality_loss_compute(sum_loss, total_params)


class ImportanceMinimalityLoss(Metric):
    """L_p loss on the sum of CI values.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.

    Args:
        pnorm: The p value for the L_p norm
        p_anneal_start_frac: The fraction of training after which to start annealing p
            (1.0 = no annealing)
        p_anneal_final_p: The final p value to anneal to (None = no annealing)
        p_anneal_end_frac: The fraction of training when annealing ends. We stay at the final p
            value from this point onward (default 1.0 = anneal until end)
        eps: The epsilon value for numerical stability.
    """

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pnorm: float,
        p_anneal_start_frac: float = 1.0,
        p_anneal_final_p: float | None = None,
        p_anneal_end_frac: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        self.pnorm = pnorm
        self.eps = eps
        self.p_anneal_start_frac = p_anneal_start_frac
        self.p_anneal_final_p = p_anneal_final_p if p_anneal_final_p is not None else None
        self.p_anneal_end_frac = p_anneal_end_frac
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
            p_anneal_start_frac=self.p_anneal_start_frac,
            p_anneal_final_p=self.p_anneal_final_p,
            p_anneal_end_frac=self.p_anneal_end_frac,
        )
        self.sum_loss += sum_loss
        self.n_examples += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _importance_minimality_loss_compute(sum_loss, n_examples)
