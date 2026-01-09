import fnmatch
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
    ModulePatternInfoConfig,
    NeuronSparsityLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SamplingType,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
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
    unmasked_recon_loss,
)
from spd.models.component_model import CIOutputs, ComponentModel


def neuron_sparsity_loss(
    model: ComponentModel,
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    module_info: list[ModulePatternInfoConfig],
    pnorm: float,
) -> Float[Tensor, ""]:
    """Compute neuron sparsity loss for modules marked as sparse.

    For each module with sparse=True in module_info:
    1. Take absolute value of each entry in the U matrix
    2. Raise to pnorm power
    3. Sum across d_out dimension
    4. Weight by causal importances
    5. Sum over C, batch, and sequence dimensions

    Args:
        model: The component model containing U matrices
        ci_upper_leaky: Causal importances for each layer (shape: ... C)
        module_info: List of module pattern configs with sparse flags
        pnorm: The p value for the Lp norm

    Returns:
        Scalar tensor with the total neuron sparsity loss
    """
    # Build set of module paths that have sparse=True
    sparse_patterns = [info.module_pattern for info in module_info if info.sparse]

    if not sparse_patterns:
        # No sparse modules, return zero loss
        device = next(iter(ci_upper_leaky.values())).device
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=next(iter(ci_upper_leaky.values())).device)

    for module_path in model.components:
        # Check if this module matches any sparse pattern
        is_sparse = any(fnmatch.fnmatch(module_path, pattern) for pattern in sparse_patterns)
        if not is_sparse:
            continue

        # Get U matrix for this module (shape: C x d_out)
        U = model.components[module_path].U

        # Take absolute value and raise to pnorm power
        U_abs_p = torch.abs(U) ** pnorm  # Shape: C x d_out

        # Sum across d_out dimension
        U_sum_d_out = U_abs_p.sum(dim=-1)  # Shape: C

        # Get causal importances for this module (shape: ... C)
        ci = ci_upper_leaky[module_path]

        # Weight by causal importances (broadcasts U_sum_d_out to ... C)
        weighted = ci * U_sum_d_out  # Shape: ... C

        # Sum over all dimensions (C, batch, sequence if present)
        module_loss = weighted.sum()

        total_loss = total_loss + module_loss

    return total_loss


def compute_total_loss(
    loss_metric_configs: list[LossMetricConfigType],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    ci: CIOutputs,
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    current_frac_of_training: float,
    sampling: SamplingType,
    use_delta_component: bool,
    n_mask_samples: int,
    output_loss_type: Literal["mse", "kl", "mem"],
    module_info: list[ModulePatternInfoConfig] | None = None,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict). terms_dict contains raw per-term values (no coeffs) and a weighted total.
    """
    total = torch.tensor(0.0, device=batch.device)
    terms: dict[str, float] = {}

    for cfg in loss_metric_configs:
        assert cfg.coeff is not None, "All loss metric configs must have a coeff"
        match cfg:
            case FaithfulnessLossConfig():
                loss = faithfulness_loss(weight_deltas=weight_deltas)
            case ImportanceMinimalityLossConfig():
                loss = importance_minimality_loss(
                    ci_upper_leaky=ci.upper_leaky,
                    current_frac_of_training=current_frac_of_training,
                    pnorm=cfg.pnorm,
                    pnorm_2=cfg.pnorm_2,
                    eps=cfg.eps,
                    p_anneal_start_frac=cfg.p_anneal_start_frac,
                    p_anneal_final_p=cfg.p_anneal_final_p,
                    p_anneal_end_frac=cfg.p_anneal_end_frac,
                )
            case NeuronSparsityLossConfig():
                assert module_info is not None, (
                    "module_info is required for NeuronSparsityLoss"
                )
                loss = neuron_sparsity_loss(
                    model=model,
                    ci_upper_leaky=ci.upper_leaky,
                    module_info=module_info,
                    pnorm=cfg.pnorm,
                )
            case UnmaskedReconLossConfig():
                loss = unmasked_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                )
            case CIMaskedReconSubsetLossConfig():
                loss = ci_masked_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    routing=cfg.routing,
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
                    routing=cfg.routing,
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
                    routing=cfg.routing,
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

        total = total + cfg.coeff * loss

    terms["loss/total"] = total.item()

    return total, terms
