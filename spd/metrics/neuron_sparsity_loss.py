import fnmatch
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import ModulePatternInfoConfig
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce


def neuron_sparsity_loss(
    model: ComponentModel,
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    module_info: list[ModulePatternInfoConfig],
    pnorm: float,
) -> Float[Tensor, ""]:
    """Compute neuron sparsity loss for modules marked as sparse.

    For each module with sparse=True in module_info:
    1. Take absolute value of each entry in the U matrix
    2. Raise to pnorm power
    3. Sum across d_out dimension
    4. Weight by causal importances
    5. Sum over C, batch, and sequence dimensions

    Args:
        model: The component model containing U matrices
        ci_upper_leaky: Causal importances for each layer (shape: ... C)
        module_info: List of module pattern configs with sparse flags
        pnorm: The p value for the Lp norm

    Returns:
        Scalar tensor with the total neuron sparsity loss
    """
    # Build set of module paths that have sparse=True
    sparse_patterns = [info.module_pattern for info in module_info if info.sparse]

    if not sparse_patterns:
        # No sparse modules, return zero loss
        device = next(iter(ci_upper_leaky.values())).device
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=next(iter(ci_upper_leaky.values())).device)

    for module_path in model.components:
        # Check if this module matches any sparse pattern
        is_sparse = any(fnmatch.fnmatch(module_path, pattern) for pattern in sparse_patterns)
        if not is_sparse:
            continue

        # Get U matrix for this module (shape: C x d_out)
        U = model.components[module_path].U

        # Take absolute value and raise to pnorm power
        U_abs_p = torch.abs(U) ** pnorm  # Shape: C x d_out

        # Sum across d_out dimension
        U_sum_d_out = U_abs_p.sum(dim=-1)  # Shape: C

        # Get causal importances for this module (shape: ... C)
        ci = ci_upper_leaky[module_path]

        # Weight by causal importances (broadcasts U_sum_d_out to ... C)
        weighted = ci * U_sum_d_out  # Shape: ... C

        # Sum over all dimensions (C, batch, sequence if present)
        module_loss = weighted.sum()

        total_loss = total_loss + module_loss

    return total_loss


def _neuron_sparsity_loss_update(
    model: ComponentModel,
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    sparse_patterns: list[str],
    pnorm: float,
) -> tuple[Float[Tensor, ""], int]:
    """Compute neuron sparsity loss for one batch.

    Returns the sum of the loss and the number of batch/seq elements for averaging.
    """
    if not sparse_patterns:
        device = next(iter(ci_upper_leaky.values())).device
        return torch.tensor(0.0, device=device), 0

    total_loss = torch.tensor(0.0, device=next(iter(ci_upper_leaky.values())).device)

    for module_path in model.components:
        # Check if this module matches any sparse pattern
        is_sparse = any(fnmatch.fnmatch(module_path, pattern) for pattern in sparse_patterns)
        if not is_sparse:
            continue

        # Get U matrix for this module (shape: C x d_out)
        U = model.components[module_path].U

        # Take absolute value and raise to pnorm power
        U_abs_p = torch.abs(U) ** pnorm  # Shape: C x d_out

        # Sum across d_out dimension
        U_sum_d_out = U_abs_p.sum(dim=-1)  # Shape: C

        # Get causal importances for this module (shape: ... C)
        ci = ci_upper_leaky[module_path]

        # Weight by causal importances (broadcasts U_sum_d_out to ... C)
        weighted = ci * U_sum_d_out  # Shape: ... C

        # Sum over all dimensions (C, batch, sequence if present)
        module_loss = weighted.sum()

        total_loss = total_loss + module_loss

    # n_examples is the number of batch/seq elements
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return total_loss, n_examples


class NeuronSparsityLoss(Metric):
    """Neuron sparsity loss on U matrices of components marked as sparse.

    This loss computes the Lp norm of U matrix entries, weighted by causal importances.
    Only applied to modules with sparse=True in their module_info config.

    Args:
        model: The component model
        device: Device to use
        module_info: List of module pattern configs with sparse flags
        pnorm: The p value for the Lp norm
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        module_info: list[ModulePatternInfoConfig],
        pnorm: float,
    ) -> None:
        self.model = model
        self.device = device
        self.pnorm = pnorm
        self.sparse_patterns = [info.module_pattern for info in module_info if info.sparse]
        self.total_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        batch_loss, n_examples = _neuron_sparsity_loss_update(
            model=self.model,
            ci_upper_leaky=ci.upper_leaky,
            sparse_patterns=self.sparse_patterns,
            pnorm=self.pnorm,
        )
        self.total_loss += batch_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        # All-reduce across distributed workers
        total_loss = all_reduce(self.total_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        if n_examples == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / n_examples
