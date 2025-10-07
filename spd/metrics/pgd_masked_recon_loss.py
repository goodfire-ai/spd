from typing import Any, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.metrics.pgd_utils import (
    PGDInitStrategy,
    optimize_adversarial_stochastic_masks,
)
from spd.models.component_model import ComponentModel
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.component_utils import (
    WeightDeltaSamplingData,
    calc_stochastic_component_mask_info,
)
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _pgd_recon_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
) -> tuple[Float[Tensor, ""], int]:
    def objective(
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor:
        # Stochastic reconstruction loss (all components at once)
        weight_deltas_and_mask_values: (
            tuple[dict[str, Float[Tensor, "d_out d_in"]], WeightDeltaSamplingData] | None
        )
        if weight_delta_mask is not None:
            assert weight_deltas is not None
            weight_deltas_and_mask_values = (weight_deltas, ("given", weight_delta_mask))
        else:
            assert weight_deltas is None
            weight_deltas_and_mask_values = None

        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=("given", component_mask),
            weight_deltas_and_mask_sampling=weight_deltas_and_mask_values,
            routing="all",
        )
        out = model(batch, mask_infos=mask_infos)
        loss_type = output_loss_type
        loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
        n_examples = out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
        return loss / n_examples

    component_masks, weight_delta_masks = optimize_adversarial_stochastic_masks(
        model=model,
        init=init,
        step_size=step_size,
        n_steps=n_steps,
        objective=objective,
        causal_importances=ci,
        weight_deltas=weight_deltas,
    )

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_delta_masks is not None:
        assert weight_deltas is not None
        weight_deltas_and_masks = {
            layer: (weight_deltas[layer], weight_delta_masks[layer]) for layer in weight_delta_masks
        }

    mask_infos = make_mask_infos(
        component_masks=component_masks,
        routing_masks="all",
        weight_deltas_and_masks=weight_deltas_and_masks,
    )

    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    return loss, out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()


def _pgd_recon_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def pgd_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    # TODO: nice args order
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _pgd_recon_loss_update(
        model=model,
        init=init,
        ci=ci,
        weight_deltas=weight_deltas,
        step_size=step_size,
        n_steps=n_steps,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
    )
    return _pgd_recon_loss_compute(sum_loss, n_examples)


class PGDReconLoss(Metric):
    """Recon loss when masking with raw CI values and routing to subsets of component layers."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        init: PGDInitStrategy,
        step_size: float,
        n_steps: int,
    ) -> None:
        self.model = model
        self.init: PGDInitStrategy = init
        self.step_size: float = step_size
        self.n_steps: int = n_steps
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "... C"]],
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _pgd_recon_loss_update(
            model=self.model,
            init=self.init,
            ci=ci,
            weight_deltas=weight_deltas,
            step_size=self.step_size,
            n_steps=self.n_steps,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _pgd_recon_loss_compute(sum_loss, n_examples)
