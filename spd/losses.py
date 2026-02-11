from typing import Any

from jaxtyping import Float
from torch import Tensor

from spd.configs import (
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossMetricConfigType,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
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
from spd.models.batch_and_loss_fns import ReconstructionLoss, recon_loss_kl
from spd.models.component_model import CIOutputs, ComponentModel
from spd.persistent_pgd import PPGDSources, persistent_pgd_recon_loss


def compute_losses(
    loss_metric_configs: list[LossMetricConfigType],
    model: ComponentModel,
    batch: Any,
    ci: CIOutputs,
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    current_frac_of_training: float,
    sampling: SamplingType,
    use_delta_component: bool,
    n_mask_samples: int,
    ppgd_sourcess: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PPGDSources
    ],
    reconstruction_loss: ReconstructionLoss,
) -> dict[LossMetricConfigType, Float[Tensor, ""]]:
    """Compute losses for each config and return a dict mapping config to loss tensor."""
    losses: dict[LossMetricConfigType, Float[Tensor, ""]] = {}

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
                    beta=cfg.beta,
                    eps=cfg.eps,
                    p_anneal_start_frac=cfg.p_anneal_start_frac,
                    p_anneal_final_p=cfg.p_anneal_final_p,
                    p_anneal_end_frac=cfg.p_anneal_end_frac,
                )
            case UnmaskedReconLossConfig():
                loss = unmasked_recon_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                )
            case CIMaskedReconSubsetLossConfig():
                loss = ci_masked_recon_subset_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    routing=cfg.routing,
                )
            case CIMaskedReconLayerwiseLossConfig():
                loss = ci_masked_recon_layerwise_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                )
            case CIMaskedReconLossConfig():
                loss = ci_masked_recon_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                )
            case StochasticReconLayerwiseLossConfig():
                loss = stochastic_recon_layerwise_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    reconstruction_loss=reconstruction_loss,
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
                    reconstruction_loss=reconstruction_loss,
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
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    routing=cfg.routing,
                )
            case PGDReconLossConfig():
                loss = pgd_recon_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                )
            case PGDReconSubsetLossConfig():
                loss = pgd_recon_subset_loss(
                    model=model,
                    reconstruction_loss=reconstruction_loss,
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
                    reconstruction_loss=reconstruction_loss,
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
            case PersistentPGDReconLossConfig() | PersistentPGDReconSubsetLossConfig():
                ppgd_sources = ppgd_sourcess[cfg]
                loss = persistent_pgd_recon_loss(
                    model=model,
                    batch=batch,
                    ppgd_sources=ppgd_sources,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    target_out=target_out,
                    output_loss_type="kl" if reconstruction_loss is recon_loss_kl else "mse",
                )

        losses[cfg] = loss

    return losses
