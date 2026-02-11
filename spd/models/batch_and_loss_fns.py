"""Reconstruction loss functions for different model types.

These functions parameterize training for different target model architectures.
"""

from typing import Any, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class ReconstructionLoss(Protocol):
    """Protocol for computing reconstruction loss between predictions and targets."""

    def __call__(self, pred: Any, target: Any) -> tuple[Float[Tensor, ""], int]: ...


def recon_loss_mse(
    pred: Float[Tensor, "... d"],
    target: Float[Tensor, "... d"],
) -> tuple[Float[Tensor, ""], int]:
    """MSE reconstruction loss. Returns (sum_of_squared_errors, n_elements)."""
    assert pred.shape == target.shape
    squared_errors = (pred - target) ** 2
    return squared_errors.sum(), pred.numel()


def recon_loss_kl(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    """KL divergence reconstruction loss for logits. Returns (sum_of_kl, n_positions)."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl_per_position = F.kl_div(log_q, p, reduction="none").sum(dim=-1)  # P · (log P − log Q)
    return kl_per_position.sum(), pred[..., 0].numel()
