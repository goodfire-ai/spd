from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _tanh_importance_minimality_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    scale: float,
    sharpness: float,
) -> tuple[Float[Tensor, ""], int]:
    """Calculate the sum of the tanh importance minimality loss over all layers.

    Uses the penalty (scale/sharpness) * tanh(x * sharpness) from Hurley & Rickard 2008.

    For small x: tanh(x*B) ≈ x*B, so penalty ≈ (A/B) * x * B = A * x (L1-like)
    For large x: tanh(x*B) → 1, so penalty → A/B (constant, no shrinking incentive)

    Args:
        ci_upper_leaky: Dictionary of CI values per layer
        scale: The A parameter - controls overall scale and gradient for small values
        sharpness: The B parameter - controls transition sharpness (larger = sharper cutoff)

    Returns:
        Tuple of (sum of tanh penalty, number of parameters)
    """
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    device = get_obj_device(ci_upper_leaky)
    sum_loss = torch.tensor(0.0, device=device)

    # Precompute (A/B) factor
    scale_factor = scale / sharpness

    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # layer_ci_upper_leaky is already >= 0
        # Penalty: (A/B) * tanh(x * B)
        penalty = scale_factor * torch.tanh(layer_ci_upper_leaky * sharpness)
        sum_loss += penalty.sum()

    n_params = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return sum_loss, n_params


def _tanh_importance_minimality_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def tanh_importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    scale: float,
    sharpness: float,
) -> Float[Tensor, ""]:
    """Compute tanh importance minimality loss.

    Uses the penalty (scale/sharpness) * tanh(x * sharpness) following Anthropic's
    sparse coding work (based on Hurley & Rickard 2008).

    Args:
        ci_upper_leaky: Dictionary of CI values per layer
        scale: The A parameter - controls overall scale and gradient for small values
        sharpness: The B parameter - controls transition sharpness (larger = sharper cutoff)

    Returns:
        Mean tanh penalty across all CI values
    """
    sum_loss, total_params = _tanh_importance_minimality_loss_update(
        ci_upper_leaky=ci_upper_leaky,
        scale=scale,
        sharpness=sharpness,
    )
    return _tanh_importance_minimality_loss_compute(sum_loss, total_params)


class TanhImportanceMinimalityLoss(Metric):
    """Tanh penalty for importance minimality, following Anthropic's sparse coding work.

    Uses the penalty (A/B) * tanh(x * B) from Hurley & Rickard 2008.

    For small x: provides L1-like gradient (penalty ≈ A*x)
    For large x: penalty saturates to A/B (no incentive to shrink)

    This penalty was found to be a Pareto improvement in L0 vs loss recovered space,
    though Anthropic noted the resulting features were harder to interpret.

    Args:
        scale: The A parameter controlling overall scale and L1-like gradient for small values.
        sharpness: The B parameter controlling transition sharpness. Larger = sharper cutoff.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        scale: float = 1.0,
        sharpness: float = 1.0,
    ) -> None:
        self.scale = scale
        self.sharpness = sharpness
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        sum_loss, total_params = _tanh_importance_minimality_loss_update(
            ci_upper_leaky=ci.upper_leaky,
            scale=self.scale,
            sharpness=self.sharpness,
        )
        self.sum_loss += sum_loss
        self.n_examples += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _tanh_importance_minimality_loss_compute(sum_loss, n_examples)
