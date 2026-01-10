import fnmatch
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.components import Components
from spd.utils.distributed_utils import all_reduce


def _filter_components_by_patterns(
    components: dict[str, Components],
    module_patterns: list[str] | None,
) -> dict[str, Components]:
    """Filter components dict by module patterns.

    Args:
        components: Dictionary mapping module paths to Components
        module_patterns: fnmatch-style patterns to filter by, or None for all modules

    Returns:
        Filtered components dictionary
    """
    if module_patterns is None:
        return components

    filtered = {}
    for module_name, comps in components.items():
        if any(fnmatch.fnmatch(module_name, pattern) for pattern in module_patterns):
            filtered[module_name] = comps

    return filtered


def _neuron_parameter_lp_loss_update(
    components: dict[str, Components],
    p: float,
    module_patterns: list[str] | None,
) -> tuple[Float[Tensor, ""], int]:
    """Compute sum of scaled u-vector L_p norms.

    For each component c: penalty = ||scale_c * u_c||_p^p where scale_c = ||v_c||_2

    Args:
        components: Dictionary mapping module paths to Components
        p: L_p norm exponent
        module_patterns: fnmatch-style patterns to filter modules, or None for all

    Returns:
        Tuple of (sum_loss, total_components)
    """
    filtered_components = _filter_components_by_patterns(components, module_patterns)
    assert filtered_components, f"No modules matched patterns: {module_patterns}"

    device = next(iter(filtered_components.values())).V.device
    sum_loss = torch.tensor(0.0, device=device)
    total_components = 0

    for comps in filtered_components.values():
        # V: [d_in, C], U: [C, d_out]
        V = comps.V
        U = comps.U

        # Compute scale_c = ||v_c||_2 for each component (columns of V)
        v_norms = torch.linalg.vector_norm(V, ord=2, dim=0)  # shape: (C,)

        # Scale each u-vector: scale_c * u_c (rows of U)
        scaled_u = v_norms.unsqueeze(1) * U  # shape: (C, d_out)

        # Compute L_p penalty: ||scale_c * u_c||_p^p for each component
        u_lp_norms = torch.linalg.vector_norm(scaled_u, ord=p, dim=1)  # shape: (C,)
        sum_loss += (u_lp_norms**p).sum()
        total_components += comps.C

    return sum_loss, total_components


def _neuron_parameter_lp_loss_compute(
    sum_loss: Float[Tensor, ""], total_components: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    """Normalize loss by number of components."""
    assert total_components > 0, "total_components must be positive"
    return sum_loss / total_components


def neuron_parameter_lp_loss(
    model: ComponentModel,
    p: float,
    module_patterns: list[str] | None = None,
) -> Float[Tensor, ""]:
    """Compute L_p penalty on scaled u-vectors across components.

    For each component c in each filtered module:
        scale_c = ||v_c||_2 (L2 norm of v-vector)
        penalty = ||scale_c * u_c||_p^p (L_p norm of scaled u-vector, raised to power p)

    This encourages sparsity in the effective parameter space by penalizing the magnitude
    of neuron (u) vectors scaled by their corresponding feature (v) vector norms.

    Args:
        model: ComponentModel containing components to penalize
        p: L_p norm exponent (e.g., 1.0 for L1, 2.0 for L2)
        module_patterns: Optional fnmatch-style patterns to filter which modules the loss
            applies to. If None (default), applies to all modules.

    Returns:
        Scalar loss tensor, averaged over number of components
    """
    sum_loss, total_components = _neuron_parameter_lp_loss_update(
        model.components, p, module_patterns
    )
    return _neuron_parameter_lp_loss_compute(sum_loss, total_components)


class NeuronParameterLpLoss(Metric):
    """L_p norm penalty on scaled u-vectors in component decompositions.

    For each component c: penalty = ||scale_c * u_c||_p^p where scale_c = ||v_c||_2

    This encourages sparsity in the effective parameter space by penalizing the magnitude
    of neuron (u) vectors scaled by their corresponding feature (v) vector norms.

    Args:
        model: ComponentModel containing components
        device: Device for accumulation tensors
        p: L_p norm exponent (e.g., 1.0 for L1, 2.0 for L2)
        module_patterns: Optional fnmatch-style patterns to filter which modules the loss
            applies to. If None (default), applies to all modules.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        p: float,
        module_patterns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.p = p
        self.module_patterns = module_patterns
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_components = torch.tensor(0, device=device)

    @override
    def update(self, **_: Any) -> None:
        """Update accumulated loss. Ignores all kwargs (no batch data needed)."""
        sum_loss, total_components = _neuron_parameter_lp_loss_update(
            self.model.components, self.p, self.module_patterns
        )
        self.sum_loss += sum_loss
        self.total_components += total_components

    @override
    def compute(self) -> Float[Tensor, ""]:
        """Compute final averaged loss, synchronized across distributed processes."""
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        total_components = all_reduce(self.total_components, op=ReduceOp.SUM)
        return _neuron_parameter_lp_loss_compute(sum_loss, total_components)
