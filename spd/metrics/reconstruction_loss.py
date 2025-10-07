from typing import Literal

from jaxtyping import Float, Int
from torch import Tensor

from spd.metrics.pgd_utils import PGDInitStrategy, optimize_adversarial_stochastic_masks
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import WeightDeltaSamplingData, calc_stochastic_component_mask_info
from spd.utils.general_utils import calc_sum_recon_loss_lm


def pgd_recon_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: Literal["all", "uniform_k-stochastic"],
) -> tuple[Float[Tensor, ""], int]:
    def objective(
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor:
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
            routing=routing,
        )
        out = model(batch, mask_infos=mask_infos)
        loss_type = output_loss_type
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
        n_examples = out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
        return total_loss / n_examples

    component_masks, weight_delta_masks = optimize_adversarial_stochastic_masks(
        model=model,
        init=init,
        step_size=step_size,
        n_steps=n_steps,
        objective=objective,
        causal_importances=ci,
        weight_deltas=weight_deltas,
    )

    weight_deltas_and_mask_sampling: (
        tuple[dict[str, Float[Tensor, " d_out d_in"]], WeightDeltaSamplingData] | None
    ) = None
    if weight_delta_masks is not None:
        assert weight_deltas is not None
        weight_deltas_and_mask_sampling = (
            weight_deltas,
            ("given", weight_delta_masks),
        )
    else:
        assert weight_deltas is None

    mask_infos = calc_stochastic_component_mask_info(
        causal_importances=ci,
        component_mask_sampling=("given", component_masks),
        weight_deltas_and_mask_sampling=weight_deltas_and_mask_sampling,
        routing=routing,
    )
    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    return loss, out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
