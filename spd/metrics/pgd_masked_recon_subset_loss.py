from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import DensityBasedRouting, PGDConfig, SubsetRoutingType
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.scheduling import get_coeff_value
from spd.utils.distributed_utils import all_reduce


def pgd_recon_subset_loss(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
    routing: SubsetRoutingType,
    current_frac_of_training: float,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = pgd_masked_recon_loss_update(
        model=model,
        ci=ci,
        weight_deltas=weight_deltas,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        pgd_config=pgd_config,
        routing=get_coeff_value(routing.routing_density, current_frac_of_training)
        if isinstance(routing, DensityBasedRouting)
        else routing,
    )
    return sum_loss / n_examples


class PGDReconSubsetLoss(Metric):
    """Recon loss when masking with adversarially-optimized values and routing to subsets of
    component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        use_delta_component: bool,
        pgd_config: PGDConfig,
        routing: SubsetRoutingType,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.use_delta_component: bool = use_delta_component
        self.routing: SubsetRoutingType = routing

        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        current_frac_of_training: float,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = pgd_masked_recon_loss_update(
            model=self.model,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            pgd_config=self.pgd_config,
            routing=get_coeff_value(self.routing.routing_density, current_frac_of_training)
            if isinstance(self.routing, DensityBasedRouting)
            else self.routing,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
