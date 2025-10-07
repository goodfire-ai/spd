from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import (
    FaithfulnessLossTrainConfig,
    ImportanceMinimalityLossTrainConfig,
    ReconstructionLossConfig,
    TrainMetricConfigType,
)
from spd.metrics.faithfulness_loss import faithfulness_loss
from spd.metrics.importance_minimality_loss import importance_minimality_loss
from spd.metrics.reconstruction_loss import reconstruction_loss
from spd.models.component_model import ComponentModel


def compute_total_loss(
    loss_metric_configs: list[TrainMetricConfigType],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    ci: dict[str, Float[Tensor, "batch C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    current_frac_of_training: float,
    sampling: Literal["continuous", "binomial"],
    use_delta_component: bool,
    n_mask_samples: int,
    output_loss_type: Literal["mse", "kl"],
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict). terms_dict contains raw per-term values (no coeffs) and a weighted total.
    """
    total = torch.tensor(0.0, device=batch.device)
    terms: dict[str, float] = {}

    for cfg in loss_metric_configs:
        match cfg:
            case ImportanceMinimalityLossTrainConfig():
                loss = importance_minimality_loss(
                    ci_upper_leaky=ci_upper_leaky,
                    current_frac_of_training=current_frac_of_training,
                    pnorm=cfg.pnorm,
                    eps=cfg.eps,
                    p_anneal_start_frac=cfg.p_anneal_start_frac,
                    p_anneal_final_p=cfg.p_anneal_final_p,
                    p_anneal_end_frac=cfg.p_anneal_end_frac,
                )
            case FaithfulnessLossTrainConfig():
                loss = faithfulness_loss(weight_deltas=weight_deltas)
            case ReconstructionLossConfig():
                loss = reconstruction_loss(
                    masking_cfg=cfg.masking,
                    routing_cfg=cfg.routing,
                    output_loss_type=output_loss_type,
                    use_delta_component=use_delta_component,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    model=model,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )

        terms[cfg.classname] = loss.item()
        total = total + cfg.coeff * loss

    terms["total"] = total.item()

    return total, terms
