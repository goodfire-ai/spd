from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import (
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossMetricConfigType,
    PGDArbHiddenActsReconLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SamplingType,
    StochasticArbHiddenActsReconLossConfig,
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
    pgd_arb_hidden_acts_recon_loss,
    pgd_recon_layerwise_loss,
    pgd_recon_loss,
    pgd_recon_subset_loss,
    stochastic_arb_hidden_acts_recon_loss,
    stochastic_hidden_acts_recon_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
)
from spd.models.component_model import CIOutputs, ComponentModel
from spd.scheduling import get_coeff_value


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
                    routing=cfg.subset_routing_cfg,
                    current_frac_of_training=current_frac_of_training,
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
                    routing=cfg.subset_routing_cfg,
                    current_frac_of_training=current_frac_of_training,
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
                    routing=cfg.subset_routing_cfg,
                    current_frac_of_training=current_frac_of_training,
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
            case StochasticArbHiddenActsReconLossConfig():
                loss = stochastic_arb_hidden_acts_recon_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    batch=batch,
                    ci=ci.lower_leaky,
                    pre_target_module_patterns=cfg.pre_target_module_patterns,
                    post_target_module_patterns=cfg.post_target_module_patterns,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )
            case PGDArbHiddenActsReconLossConfig():
                loss = pgd_arb_hidden_acts_recon_loss(
                    model=model,
                    batch=batch,
                    ci=ci.lower_leaky,
                    post_target_module_path=cfg.post_target_module_path,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                )

        if isinstance(loss, dict):
            for key, value in loss.items():
                terms[f"loss/{key}"] = value.item()
                coeff = get_coeff_value(cfg.coeff, current_frac_of_training)
                total = total + coeff * value
        else:
            terms[f"loss/{cfg.classname}"] = loss.item()
            coeff = get_coeff_value(cfg.coeff, current_frac_of_training)
            total = total + coeff * loss

    terms["loss/total"] = total.item()

    return total, terms
