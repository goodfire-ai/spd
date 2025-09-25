from typing import Any, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel


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
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    pnorm = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
    )
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
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    eps: float,
    current_frac_of_training: float,
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


    Args:
        pnorm: The p value for the L_p norm
        p_anneal_start_frac: The fraction of training after which to start annealing p
            (1.0 = no annealing)
        p_anneal_final_p: The final p value to anneal to (None = no annealing)
        p_anneal_end_frac: The fraction of training when annealing ends. We stay at the final p
            value from this point onward (default 1.0 = anneal until end)
        eps: The epsilon value for numerical stability.
    """

    slow = False
    is_differentiable: bool | None = True
    full_state_update: bool | None = False  # Avoid double update calls

    sum_loss: Float[Tensor, " C"]
    n_examples: Int[Tensor, ""]

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        pnorm: Any,
        p_anneal_start_frac: float = 1.0,
        p_anneal_final_p: float | None = None,
        p_anneal_end_frac: float = 1.0,
        eps: Any = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pnorm = float(pnorm)
        self.eps = float(eps)
        self.p_anneal_start_frac = float(p_anneal_start_frac)
        self.p_anneal_final_p = float(p_anneal_final_p) if p_anneal_final_p is not None else None
        self.p_anneal_end_frac = float(p_anneal_end_frac)
        self.add_state("sum_loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        *,
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        current_frac_of_training: float,
        **kwargs: Any,
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
        return _importance_minimality_loss_compute(self.sum_loss, self.n_examples)
