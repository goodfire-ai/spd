from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import (
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    CosineSchedule,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LinearSchedule,
    LossMetricConfigType,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SamplingType,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
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
    stochastic_hidden_acts_recon_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
)
from spd.models.component_model import CIOutputs, ComponentModel
from spd.scheduling import get_cosine_schedule_value, get_linear_schedule_value


def get_loss_coeff(
    coeff: LinearSchedule | CosineSchedule | float | int,
    current_frac_of_training: float,
) -> float:
    match coeff:
        case LinearSchedule():
            return get_linear_schedule_value(coeff, current_frac_of_training)
        case CosineSchedule():
            return get_cosine_schedule_value(coeff, current_frac_of_training)
        case float() | int():
            return coeff


def compute_total_loss(
    loss_metric_configs: list[LossMetricConfigType],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    ci: CIOutputs,
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    current_frac_of_training: float,
    sampling: SamplingType,
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
        assert cfg.coeff is not None, "All loss metric configs must have a coeff"
        match cfg:
            case ImportanceMinimalityLossConfig():
                loss = importance_minimality_loss(
                    ci_upper_leaky=ci.upper_leaky,
                    current_frac_of_training=current_frac_of_training,
                    pnorm=cfg.pnorm,
                    eps=cfg.eps,
                    p_anneal_start_frac=cfg.p_anneal_start_frac,
                    p_anneal_final_p=cfg.p_anneal_final_p,
                    p_anneal_end_frac=cfg.p_anneal_end_frac,
                )
            case CIMaskedReconSubsetLossConfig():
                loss = ci_masked_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                )
            case CIMaskedReconLayerwiseLossConfig():
                loss = ci_masked_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                )
            case CIMaskedReconLossConfig():
                loss = ci_masked_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                )
            case FaithfulnessLossConfig():
                loss = faithfulness_loss(weight_deltas=weight_deltas)
            case StochasticReconLayerwiseLossConfig():
                loss = stochastic_recon_layerwise_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )
            case StochasticReconLossConfig():
                loss = stochastic_recon_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )
            case StochasticReconSubsetLossConfig():
                loss = stochastic_recon_subset_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )
            case PGDReconLossConfig():
                loss = pgd_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                )
            case PGDReconSubsetLossConfig():
                loss = pgd_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                )
            case PGDReconLayerwiseLossConfig():
                loss = pgd_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                )
            case StochasticHiddenActsReconLossConfig():
                loss = stochastic_hidden_acts_recon_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    batch=batch,
                    pre_weight_acts=pre_weight_acts,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )

        terms[f"loss/{cfg.classname}"] = loss.item()

        coeff = get_loss_coeff(cfg.coeff, current_frac_of_training)
        total = total + coeff * loss

    terms["loss/total"] = total.item()

    return total, terms
