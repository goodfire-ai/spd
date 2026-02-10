"""Batch handling and reconstruction loss functions for different model types.

These functions parameterize ComponentModel and training for different target model architectures.
"""

from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

OutputExtract = Literal["first_element", "logits_attr"]


class RunBatch[BatchT, OutputT](Protocol):
    """Protocol for running a batch through a model and returning the output."""

    def __call__(self, model: nn.Module, batch: BatchT) -> OutputT: ...


class ReconstructionLoss[OutputT](Protocol):
    """Protocol for computing reconstruction loss between predictions and targets."""

    def __call__(self, pred: OutputT, target: OutputT) -> tuple[Float[Tensor, ""], int]: ...


def run_batch_raw(model: nn.Module, batch: Any) -> Any:
    return model(batch)


def run_batch_first_element(model: nn.Module, batch: Any) -> Tensor:
    return model(batch)[0]


def run_batch_logits_attr(model: nn.Module, batch: Any) -> Tensor:
    return model(batch).logits


def make_run_batch(output_extract: OutputExtract | None) -> RunBatch[Any, Any]:
    match output_extract:
        case None:
            return run_batch_raw
        case "first_element":
            return run_batch_first_element
        case "logits_attr":
            return run_batch_logits_attr


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
