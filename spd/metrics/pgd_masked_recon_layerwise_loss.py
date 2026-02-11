from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.batch_and_loss_fns import ReconstructionLoss
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import LayerRouter
from spd.utils.distributed_utils import all_reduce


def _pgd_recon_layerwise_loss_update(
    *,
    model: ComponentModel,
    batch: Any,
    target_out: Any,
    reconstruction_loss: ReconstructionLoss,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], Int[Tensor, ""]]:
    device = next(iter(ci.values())).device
    sum_loss = torch.tensor(0.0, device=device)
    n_examples = torch.tensor(0, device=device)
    for layer in model.target_module_paths:
        sum_loss_layer, n_examples_layer = pgd_masked_recon_loss_update(
            model=model,
            batch=batch,
            ci=ci,
            weight_deltas=weight_deltas,
            target_out=target_out,
            reconstruction_loss=reconstruction_loss,
            router=LayerRouter(device=device, layer_name=layer),
            pgd_config=pgd_config,
        )
        sum_loss += sum_loss_layer
        n_examples += n_examples_layer
    return sum_loss, n_examples


def pgd_recon_layerwise_loss(
    *,
    model: ComponentModel,
    batch: Any,
    target_out: Any,
    reconstruction_loss: ReconstructionLoss,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _pgd_recon_layerwise_loss_update(
        model=model,
        batch=batch,
        target_out=target_out,
        reconstruction_loss=reconstruction_loss,
        ci=ci,
        weight_deltas=weight_deltas,
        pgd_config=pgd_config,
    )
    return sum_loss / n_examples


class PGDReconLayerwiseLoss(Metric):
    """Recon loss when masking with adversarially-optimized values and routing to one layer at a
    time."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        reconstruction_loss: ReconstructionLoss,
        pgd_config: PGDConfig,
        device: str,
        use_delta_component: bool,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.reconstruction_loss = reconstruction_loss
        self.use_delta_component: bool = use_delta_component
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Any,
        target_out: Any,
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _pgd_recon_layerwise_loss_update(
            model=self.model,
            batch=batch,
            target_out=target_out,
            reconstruction_loss=self.reconstruction_loss,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            pgd_config=self.pgd_config,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
