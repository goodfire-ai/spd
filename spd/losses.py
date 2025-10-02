from typing import Literal, cast

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.component_utils import (
    calc_stochastic_component_mask_info,
    sample_uniform_k_subset_routing_masks,
)
from spd.utils.general_utils import calc_kl_divergence_lm


def calc_importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float, eps: float = 1e-12
) -> Float[Tensor, ""]:
    """Calculate the importance minimality loss on the upper leaky relu causal importances.

    Args:
        ci_upper_leaky: Dictionary of causal importances upper leaky relu for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.
        eps: The epsilon to add to the causal importances to avoid division by zero when computing
            the gradients for pnorm < 1.

    Returns:
        The importance minimality loss on the upper leaky relu causal importances.
    """
    total_loss = torch.zeros_like(next(iter(ci_upper_leaky.values())))

    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # Note, the paper uses an absolute value but our layer_ci_upper_leaky is already > 0
        total_loss = total_loss + (layer_ci_upper_leaky + eps) ** pnorm

    # Sum over the C dimension and mean over the other dimensions
    return total_loss.sum(dim=-1).mean()


def calc_masked_recon_layerwise_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    mask_infos_list: list[dict[str, ComponentsMaskInfo]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component layer at a time.

    This function takes the mean loss over all masks in mask_infos_list.

    Args:
        model: The component model
        batch: Input batch
        mask_infos_list: Mask infos for each stochastic source (there are config.n_mask_samples
            stochastic sources).
        target_out: Target model output
        loss_type: Type of loss to calculate
        device: Device to run computations on

    Returns:
        The recon loss
    """
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
    total_loss = torch.tensor(0.0, device=device)
    for mask_infos in mask_infos_list:
        for module_name, mask_info in mask_infos.items():
            modified_out = model(batch, mode="components", mask_infos={module_name: mask_info})
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            else:
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            total_loss += loss
    n_modified_components = len(mask_infos_list[0])
    n_stochastic_sources = len(mask_infos_list)
    return total_loss / (n_modified_components * n_stochastic_sources)


def calc_masked_recon_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    mask_infos_list: list[dict[str, ComponentsMaskInfo]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the recon loss when applying all (masked) component layers at once.

    This function takes the mean loss over all masks in mask_infos_list.

    Args:
        model: The component model
        batch: Input batch
        mask_infos_list: Mask infos for each stochastic source (there are config.n_mask_samples
            stochastic sources).
        target_out: Target model output
        loss_type: Type of loss to calculate
        device: Device to run computations on

    Returns:
        The recon loss
    """
    # Do a forward pass with all components
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"

    total_loss = torch.tensor(0.0, device=device)
    for mask_infos in mask_infos_list:
        out = model(batch, mode="components", mask_infos=mask_infos)
        if loss_type == "mse":
            loss = ((out - target_out) ** 2).mean()
        else:
            loss = calc_kl_divergence_lm(pred=out, target=target_out)
            # flat_logits = einops.rearrange(out, "b seq_len vocab -> (b seq_len) vocab")
            # masked_batch = batch.clone()
            # masked_batch[:, 0] = -100
            # flat_masked_batch = masked_batch.flatten()
            # loss = F.cross_entropy(flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100)
        total_loss += loss

    return total_loss / len(mask_infos_list)


def calc_faithfulness_loss(
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    device: str | torch.device,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component weights (V@U) and target weights.

    We sum over all layers and normalize by the number of parameters in the model (this includes any
    inserted identity matrices).
    """

    n_params = sum(param.numel() for param in weight_deltas.values())
    mse = torch.tensor(0.0, device=device)
    for param in weight_deltas.values():
        mse += ((param) ** 2).sum()
    # Normalize by the number of parameters in the model (including any inserted identity matrices)
    mse = mse / n_params
    return mse


def _optimize_adversarial_stochastic_masks(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    device: str,
) -> tuple[
    dict[str, Float[Tensor, "... C"]],
    dict[str, Float[Tensor, "..."]] | None,
]:
    """Optimize adversarial stochastic sources in [0,1] via PGD to maximize stochastic losses.

    Returns (rand_tensors, weight_delta_rand_masks) where:
    - rand_tensors: per-layer tensors shaped like corresponding causal_importances
    - weight_delta_rand_masks: per-layer tensors with leading dims matching inputs (or None)
    """
    assert config.sampling == "continuous", (
        "PGD mask optimization only supports continuous sampling"
    )

    # Initialize adversarial variables
    rand_tensors: dict[str, Float[Tensor, "... C"]] = {}
    weight_delta_rand_masks: dict[str, Float[Tensor, ...]] | None = (
        {} if weight_deltas is not None else None
    )
    leading_dims = next(iter(causal_importances.values())).shape[:-1]
    for layer, ci in causal_importances.items():
        init = torch.rand_like(ci) if config.pgd_mask_random_init else torch.full_like(ci, 0.5)
        init.requires_grad_(True)
        rand_tensors[layer] = init
        if weight_delta_rand_masks is not None:
            wd_init = (
                torch.rand(*leading_dims, device=ci.device, dtype=ci.dtype)
                if config.pgd_mask_random_init
                else torch.full(leading_dims, 0.5, device=ci.device, dtype=ci.dtype)
            )
            wd_init.requires_grad_(True)
            weight_delta_rand_masks[layer] = wd_init

    step_size = 0.1 if config.pgd_mask_step_size is None else float(config.pgd_mask_step_size)

    def build_objective() -> Tensor:
        objective = torch.tensor(0.0, device=device)

        # Stochastic reconstruction loss (all components at once)
        if config.stochastic_recon_coeff is not None:
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=causal_importances,
                sampling=config.sampling,
                routing="all",
                weight_deltas=weight_deltas if config.use_delta_component else None,
                rand_tensors=rand_tensors,
                weight_delta_rand_mask=weight_delta_rand_masks,
            )
            loss_val = calc_masked_recon_loss(
                model=model,
                batch=batch,
                mask_infos_list=[mask_infos],
                target_out=target_out,
                loss_type=config.output_loss_type,
                device=device,
            )
            objective = objective + config.stochastic_recon_coeff * loss_val

        # Stochastic reconstruction layerwise loss
        if config.stochastic_recon_layerwise_coeff is not None:
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=causal_importances,
                sampling=config.sampling,
                routing="all",
                weight_deltas=weight_deltas if config.use_delta_component else None,
                rand_tensors=rand_tensors,
                weight_delta_rand_mask=weight_delta_rand_masks,
            )
            loss_val = calc_masked_recon_layerwise_loss(
                model=model,
                batch=batch,
                mask_infos_list=[mask_infos],
                target_out=target_out,
                loss_type=config.output_loss_type,
                device=device,
            )
            objective = objective + config.stochastic_recon_layerwise_coeff * loss_val

        # Stochastic reconstruction subset loss (routing subset)
        if config.stochastic_recon_subset_coeff is not None:
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=causal_importances,
                sampling=config.sampling,
                routing="uniform_k-stochastic",
                weight_deltas=weight_deltas if config.use_delta_component else None,
                rand_tensors=rand_tensors,
                weight_delta_rand_mask=weight_delta_rand_masks,
            )
            loss_val = calc_masked_recon_loss(
                model=model,
                batch=batch,
                mask_infos_list=[mask_infos],
                target_out=target_out,
                loss_type=config.output_loss_type,
                device=device,
            )
            objective = objective + config.stochastic_recon_subset_coeff * loss_val

        return objective

    # PGD ascent
    for _ in range(int(config.pgd_mask_steps)):
        # Zero any existing grads on rand tensors
        for v in rand_tensors.values():
            if v.grad is not None:
                v.grad = None
        if weight_delta_rand_masks is not None:
            for v in weight_delta_rand_masks.values():
                if v.grad is not None:
                    v.grad = None

        objective = build_objective()
        adv_vars = list(rand_tensors.values()) + (
            list(weight_delta_rand_masks.values()) if weight_delta_rand_masks is not None else []
        )
        raw_grads = torch.autograd.grad(
            objective,
            adv_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grads = cast(tuple[Tensor | None, ...], raw_grads)

        with torch.no_grad():
            # Update all adversarial variables in the same order they were passed to autograd
            for v, g in zip(adv_vars, grads, strict=True):
                if g is not None:
                    v.add_(step_size * g.sign())
                v.clamp_(0.0, 1.0)

    # Detach to avoid tracking grads in the outer loss backward
    with torch.no_grad():
        for layer in list(rand_tensors.keys()):
            rand_tensors[layer].detach_().clamp_(0.0, 1.0)
        if weight_delta_rand_masks is not None:
            for layer in list(weight_delta_rand_masks.keys()):
                weight_delta_rand_masks[layer].detach_().clamp_(0.0, 1.0)

    return rand_tensors, weight_delta_rand_masks


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    causal_importances: dict[str, Float[Tensor, "... C"]],
    causal_importances_upper_leaky: dict[str, Float[Tensor, "... C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    device: str,
    current_p: float | None = None,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Calculate all losses and return total loss and individual loss terms.

    Args:
        model: The component model
        batch: Input batch
        config: Configuration object with loss coefficients
        causal_importances: Causal importance masks
        causal_importances_upper_leaky: Upper leaky causal importances for regularization
        target_out: Target model output
        weight_deltas: Weight deltas between the target model and component weights (V@U)
        device: Device to run computations on
        current_p: Current p value for L_p sparsity loss (if using annealing)
    Returns:
        Tuple of (total_loss, loss_terms_dict)
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_terms: dict[str, float] = {}

    # Faithfulness loss
    if config.faithfulness_coeff is not None:
        faithfulness_loss = calc_faithfulness_loss(weight_deltas, device)
        total_loss += config.faithfulness_coeff * faithfulness_loss
        loss_terms["faithfulness"] = faithfulness_loss.item()

    # CI reconstruction loss
    if config.ci_recon_coeff is not None:
        ci_recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=[make_mask_infos(causal_importances, weight_deltas_and_masks=None)],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.ci_recon_coeff * ci_recon_loss
        loss_terms["ci_recon"] = ci_recon_loss.item()

    # Prepare adversarial rand tensors if enabled
    use_pgd_masks = (
        config.pgd_mask_enabled
        and config.sampling == "continuous"
        and config.pgd_mask_steps > 0
        and (
            config.stochastic_recon_coeff is not None
            or config.stochastic_recon_layerwise_coeff is not None
            or config.stochastic_recon_subset_coeff is not None
        )
    )

    adv_rand_tensors: dict[str, Float[Tensor, "... C"]] | None = None
    adv_weight_delta_rand_masks: dict[str, Float[Tensor, ...]] | None = None
    if use_pgd_masks:
        adv_rand_tensors, adv_weight_delta_rand_masks = _optimize_adversarial_stochastic_masks(
            model=model,
            batch=batch,
            config=config,
            causal_importances=causal_importances,
            target_out=target_out,
            weight_deltas=weight_deltas if config.use_delta_component else None,
            device=device,
        )

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        if use_pgd_masks and adv_rand_tensors is not None:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    routing="all",
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    rand_tensors=adv_rand_tensors,
                    weight_delta_rand_mask=adv_weight_delta_rand_masks,
                )
                for _ in range(config.n_mask_samples)
            ]
        else:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    routing="all",
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                )
                for _ in range(config.n_mask_samples)
            ]
        stochastic_recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=stoch_mask_infos_list,
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.stochastic_recon_coeff * stochastic_recon_loss
        loss_terms["stochastic_recon"] = stochastic_recon_loss.item()

    # CI reconstruction layerwise loss
    if config.ci_recon_layerwise_coeff is not None:
        ci_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            mask_infos_list=[make_mask_infos(causal_importances, weight_deltas_and_masks=None)],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.ci_recon_layerwise_coeff * ci_recon_layerwise_loss
        loss_terms["ci_recon_layerwise"] = ci_recon_layerwise_loss.item()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
        if use_pgd_masks and adv_rand_tensors is not None:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    routing="all",
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    rand_tensors=adv_rand_tensors,
                    weight_delta_rand_mask=adv_weight_delta_rand_masks,
                )
                for _ in range(config.n_mask_samples)
            ]
        else:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    routing="all",
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                )
                for _ in range(config.n_mask_samples)
            ]
        stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            mask_infos_list=stoch_mask_infos_list,
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
        loss_terms["stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # CI subset reconstruction loss
    if config.ci_masked_recon_subset_coeff is not None:
        subset_routing_masks = sample_uniform_k_subset_routing_masks(
            mask_shape=next(iter(causal_importances.values())).shape[:-1],
            modules=list(causal_importances.keys()),
            device=device,
        )
        mask_infos = make_mask_infos(
            component_masks=causal_importances,
            routing_masks=subset_routing_masks,
            weight_deltas_and_masks=None,
        )
        ci_recon_subset_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=[mask_infos],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.ci_masked_recon_subset_coeff * ci_recon_subset_loss
        loss_terms["ci_recon_subset"] = ci_recon_subset_loss.item()

    # Stochastic reconstruction subset loss
    if config.stochastic_recon_subset_coeff is not None:
        if use_pgd_masks and adv_rand_tensors is not None:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    routing="uniform_k-stochastic",
                    rand_tensors=adv_rand_tensors,
                    weight_delta_rand_mask=adv_weight_delta_rand_masks,
                )
                for _ in range(config.n_mask_samples)
            ]
        else:
            stoch_mask_infos_list = [
                calc_stochastic_component_mask_info(
                    causal_importances=causal_importances,
                    sampling=config.sampling,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    routing="uniform_k-stochastic",
                )
                for _ in range(config.n_mask_samples)
            ]
        stochastic_recon_subset_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=stoch_mask_infos_list,
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.stochastic_recon_subset_coeff * stochastic_recon_subset_loss
        loss_terms["stochastic_recon_subset"] = stochastic_recon_subset_loss.item()

    # Importance minimality loss
    pnorm_value = current_p if current_p is not None else config.pnorm
    importance_minimality_loss = calc_importance_minimality_loss(
        ci_upper_leaky=causal_importances_upper_leaky, pnorm=pnorm_value
    )
    total_loss += config.importance_minimality_coeff * importance_minimality_loss
    loss_terms["importance_minimality"] = importance_minimality_loss.item()

    loss_terms["total"] = total_loss.item()

    return total_loss, loss_terms
