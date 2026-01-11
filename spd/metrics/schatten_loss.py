from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import Components
from spd.utils.distributed_utils import all_reduce


def _schatten_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    components: dict[str, Components],
    pnorm: float,
) -> Float[Tensor, ""]:
    """Calculate Schatten loss on active components.

    The Schatten loss is:
        L = sum over layers of sum over components of:
            (ci_upper_leaky^pnorm · (||V_c||_2^2 + ||U_c||_2^2))

    where:
        - ci_upper_leaky is summed over batch/seq dims before weighting
        - V_c is the c-th column of V: shape [d_in]
        - U_c is the c-th row of U: shape [d_out]

    Args:
        ci_upper_leaky: Dict mapping layer names to CI values [... C]
        components: Dict mapping layer names to Components
        pnorm: Power to raise CI values to

    Returns:
        The Schatten loss as a scalar tensor.
    """
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    assert set(ci_upper_leaky.keys()) == set(components.keys()), (
        f"Keys mismatch: {set(ci_upper_leaky.keys())} vs {set(components.keys())}"
    )

    total_loss: Float[Tensor, ""] | None = None
    for layer_name, layer_ci in ci_upper_leaky.items():
        component = components[layer_name]

        # Sum CI over batch/seq dimensions, then raise to pnorm
        # layer_ci shape: [... C] -> ci_sum shape: [C]
        ci_sum = layer_ci.sum(dim=tuple(range(layer_ci.dim() - 1)))
        ci_weighted = ci_sum**pnorm

        # Compute squared L2 norms for each component
        # V shape: [d_in, C] -> V_norms shape: [C]
        V_norms = component.V.square().sum(dim=0)
        # U shape: [C, d_out] -> U_norms shape: [C]
        U_norms = component.U.square().sum(dim=1)
        schatten_norms = V_norms + U_norms

        # Weight schatten norms by CI and sum over components
        layer_loss = (ci_weighted * schatten_norms).sum()
        total_loss = layer_loss if total_loss is None else total_loss + layer_loss

    assert total_loss is not None
    return total_loss


def schatten_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    components: dict[str, Components],
    pnorm: float,
) -> Float[Tensor, ""]:
    """Calculate Schatten loss on active components.

    Args:
        ci_upper_leaky: Dict mapping layer names to CI values [... C]
        components: Dict mapping layer names to Components
        pnorm: Power to raise CI values to

    Returns:
        The Schatten loss as a scalar tensor.
    """
    return _schatten_loss_update(ci_upper_leaky, components, pnorm)


class SchattenLoss(Metric):
    """Schatten loss on the active components.

    Weights the squared L2 norms of V and U component matrices by causal importance values.

    Args:
        pnorm: Power to raise CI values to before weighting component norms.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pnorm: float,
    ) -> None:
        self.model = model
        self.pnorm = pnorm
        self.device = device
        self.accumulated_loss = torch.tensor(0.0, device=device)

    @override
    def update(
        self,
        *,
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        loss = _schatten_loss_update(
            ci_upper_leaky=ci.upper_leaky,
            components=self.model.components,
            pnorm=self.pnorm,
        )
        self.accumulated_loss += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        # Clone to avoid returning a reference to internal state
        # Without cloning, callers would get a reference to accumulated_loss,
        # which would be updated by subsequent update() calls
        return all_reduce(self.accumulated_loss.clone(), op=ReduceOp.SUM)
