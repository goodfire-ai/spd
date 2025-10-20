from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import DensityBasedRouting, SubsetRoutingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import make_mask_infos
from spd.scheduling import get_coeff_value
from spd.utils.component_utils import calc_routing_masks
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _ci_masked_recon_subset_loss_update(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    routing: SubsetRoutingType,
    current_frac_of_training: float,
) -> tuple[Float[Tensor, ""], int]:
    materialized_routing = (
        get_coeff_value(routing.k, current_frac_of_training)
        if isinstance(routing, DensityBasedRouting)
        else routing
    )

    subset_routing_masks = calc_routing_masks(
        routing=materialized_routing,
        leading_dims=next(iter(ci.values())).shape[:-1],
        module_names=list(ci.keys()),
        device=batch.device,
    )

    mask_infos = make_mask_infos(
        component_masks=ci,
        routing_masks=subset_routing_masks,
        weight_deltas_and_masks=None,
    )
    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    return loss, out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()


def _ci_masked_recon_subset_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_subset_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    routing: SubsetRoutingType,
    current_frac_of_training: float,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_subset_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        routing=routing,
        current_frac_of_training=current_frac_of_training,
    )
    return _ci_masked_recon_subset_loss_compute(sum_loss, n_examples)


class CIMaskedReconSubsetLoss(Metric):
    """Recon loss when masking with raw CI values and routing to subsets of component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        routing: SubsetRoutingType,
    ) -> None:
        self.model = model
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
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
        current_frac_of_training: float,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _ci_masked_recon_subset_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            routing=self.routing,
            current_frac_of_training=current_frac_of_training,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _ci_masked_recon_subset_loss_compute(sum_loss, n_examples)
