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
from spd.metrics import (
    ci_masked_recon_layerwise_loss,
    ci_masked_recon_loss,
    ci_masked_recon_subset_loss,
    faithfulness_loss,
    importance_minimality_loss,
    pgd_recon_layerwise_loss,
    pgd_recon_loss,
    pgd_recon_subset_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
)
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
            case CIMaskedReconSubsetLossTrainConfig():
                loss = ci_masked_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                )
            case CIMaskedReconLayerwiseLossTrainConfig():
                loss = ci_masked_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                )
            case CIMaskedReconLossTrainConfig():
                loss = ci_masked_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                )
            case FaithfulnessLossTrainConfig():
                loss = faithfulness_loss(weight_deltas=weight_deltas)
            case StochasticReconLayerwiseLossTrainConfig():
                loss = stochastic_recon_layerwise_loss(
                    model=model,
                    sampling=sampling,
                    use_delta_component=use_delta_component,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )
            case StochasticReconLossTrainConfig():
                loss = stochastic_recon_loss(
                    model=model,
                    sampling=sampling,
                    use_delta_component=use_delta_component,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )
            case StochasticReconSubsetLossTrainConfig():
                loss = stochastic_recon_subset_loss(
                    model=model,
                    sampling=sampling,
                    use_delta_component=use_delta_component,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )
            case PGDReconLossTrainConfig():
                loss = pgd_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    weight_deltas=weight_deltas,
                    init=cfg.init,
                    step_size=cfg.step_size,
                    n_steps=cfg.n_steps,
                )
            case PGDReconSubsetLossTrainConfig():
                loss = pgd_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    use_delta_component=use_delta_component,
                    weight_deltas=weight_deltas,
                    init=cfg.init,
                    step_size=cfg.step_size,
                    n_steps=cfg.n_steps,
                )
            case PGDReconLayerwiseLossTrainConfig():
                loss = pgd_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    use_delta_component=use_delta_component,
                    weight_deltas=weight_deltas,
                    init=cfg.init,
                    step_size=cfg.step_size,
                    n_steps=cfg.n_steps,
                )
        terms[cfg.classname] = loss.item()
        total = total + cfg.coeff * loss

    terms["total"] = total.item()

    return total, terms
