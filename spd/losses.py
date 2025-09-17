from typing import Literal

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.mask_info import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos
from spd.models.component_model import ComponentModel
from spd.models.components import Components, ComponentsOrModule, EmbeddingComponents
from spd.utils.component_utils import calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm


def calc_embedding_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    unembed: bool,
    device: str,
) -> Float[Tensor, ""]:
    """
    recon loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponents``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the masked embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the masked embedding output
    and the target embedding output is used as the loss.
    """

    assert len(model.components_or_modules) == 1, "Only one embedding component is supported"
    components_or_module = next(iter(model.components_or_modules.values()))
    components = components_or_module.components
    original = components_or_module.original
    assert isinstance(components, EmbeddingComponents)

    # --- original embedding output --------------------------------------------------------- #
    target_out: Float[Tensor, "... d_emb"] = original(batch)

    # --- masked embedding output ----------------------------------------------------------- #
    loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        assert len(mask_info) == 1, "Only one embedding component is supported"
        mask = next(iter(mask_info.values()))
        masked_out: Float[Tensor, "... d_emb"] = components(batch, mask=mask)

        if unembed:
            assert hasattr(model.patched_model, "lm_head"), (
                "Only supports unembedding named lm_head"
            )
            assert isinstance(model.patched_model.lm_head, nn.Module)
            target_out_unembed = model.patched_model.lm_head(target_out)
            masked_out_unembed = model.patched_model.lm_head(masked_out)
            loss += calc_kl_divergence_lm(pred=masked_out_unembed, target=target_out_unembed)
        else:
            loss += ((masked_out - target_out) ** 2).sum(dim=-1).mean()

    loss /= len(masks)

    return loss


def calc_schatten_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    components: dict[str, Components],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the Schatten loss on the active components.

    The Schatten loss is calculated as:
        L = Σ_{components} mean(ci_upper_leaky^pnorm · (||V||_2^2 + ||U||_2^2))

    where:
        - ci_upper_leaky are the upper leaky relu causal importances for each component
        - pnorm is the power to raise the mask to
        - V and U are the component matrices
        - ||·||_2 is the L2 norm

    Args:
        ci_upper_leaky: Dictionary of upper leaky relu causal importances for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.
        components: Dictionary of components for each layer.
        device: The device to compute the loss on.

    Returns:
        The Schatten loss as a scalar tensor.
    """

    total_loss = torch.tensor(0.0, device=device)
    for component_name, component in components.items():
        V_norms = component.V.square().sum(dim=-2)
        U_norms = component.U.square().sum(dim=-1)
        schatten_norms = V_norms + U_norms
        loss = einops.einsum(
            ci_upper_leaky[component_name] ** pnorm, schatten_norms, "... C, C -> ..."
        )
        total_loss += loss.mean()
    return total_loss


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
        total_loss += loss

    return total_loss / len(mask_infos_list)


def calc_weight_deltas(
    model: ComponentModel, device: str | torch.device
) -> dict[str, Float[Tensor, " d_out d_in"]]:
    """Calculate the weight differences between the target model and component weights (V@U) for
    each layer."""
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] = {}
    for comp_name, components_or_module in model.components_or_modules.items():
        assert isinstance(components_or_module, ComponentsOrModule)
        if components_or_module.components is not None:
            weight_deltas[comp_name] = (
                components_or_module.original_weight - components_or_module.components.weight
            )
        if components_or_module.identity_components is not None:
            id_name = f"identity_{comp_name}"
            id_mat = components_or_module.identity_components.weight
            weight_deltas[id_name] = (
                torch.eye(id_mat.shape[0], device=device, dtype=id_mat.dtype) - id_mat
            )
    return weight_deltas


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


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    causal_importances: dict[str, Float[Tensor, "batch C"]],
    causal_importances_upper_leaky: dict[str, Float[Tensor, "batch C"]],
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

    # Reconstruction loss
    if config.recon_coeff is not None:
        recon_mask_infos = make_mask_infos(causal_importances, None)
        recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=[recon_mask_infos],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.recon_coeff * recon_loss
        loss_terms["recon"] = recon_loss.item()

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        stochastic_masks_list = calc_stochastic_masks(
            causal_importances=causal_importances,
            n_mask_samples=config.n_mask_samples,
            sampling=config.sampling,
        )

        stoch_mask_infos_list = []
        for stochastic_masks in stochastic_masks_list:
            deltas_and_masks = (
                {
                    key: (weight_deltas[key], stochastic_masks.weight_delta_masks[key])
                    for key in weight_deltas
                }
                if config.use_delta_component
                else None
            )
            stoch_mask_infos = make_mask_infos(
                masks=stochastic_masks.component_masks,
                weight_deltas_and_masks=deltas_and_masks,
            )
            stoch_mask_infos_list.append(stoch_mask_infos)

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

    # Reconstruction layerwise loss
    if config.recon_layerwise_coeff is not None:
        recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            mask_infos_list=[make_mask_infos(causal_importances)],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.recon_layerwise_coeff * recon_layerwise_loss
        loss_terms["recon_layerwise"] = recon_layerwise_loss.item()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
        stochastic_masks_list = calc_stochastic_masks(
            causal_importances=causal_importances,
            n_mask_samples=config.n_mask_samples,
            sampling=config.sampling,
        )
        mask_infos_list: list[dict[str, ComponentsMaskInfo]] = []
        for stochastic_masks in stochastic_masks_list:
            weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = (
                {
                    key: (weight_deltas[key], stochastic_masks.weight_delta_masks[key])
                    for key in weight_deltas
                }
                if config.use_delta_component
                else None
            )
            mask_infos = make_mask_infos(
                masks=stochastic_masks.component_masks,
                weight_deltas_and_masks=weight_deltas_and_masks,
            )
            mask_infos_list.append(mask_infos)

        stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            mask_infos_list=mask_infos_list,
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
        loss_terms["stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # Importance minimality loss
    pnorm_value = current_p if current_p is not None else config.pnorm
    importance_minimality_loss = calc_importance_minimality_loss(
        ci_upper_leaky=causal_importances_upper_leaky, pnorm=pnorm_value
    )
    total_loss += config.importance_minimality_coeff * importance_minimality_loss
    loss_terms["importance_minimality"] = importance_minimality_loss.item()

    # Schatten loss
    if config.schatten_coeff is not None:
        schatten_loss = calc_schatten_loss(
            ci_upper_leaky=causal_importances_upper_leaky,
            pnorm=pnorm_value,
            components=model.components,
            device=device,
        )
        total_loss += config.schatten_coeff * schatten_loss
        loss_terms["schatten"] = schatten_loss.item()

    # Output reconstruction loss
    if config.out_recon_coeff is not None:
        masks_all_ones = {k: torch.ones_like(v) for k, v in causal_importances.items()}
        out_recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            mask_infos_list=[make_mask_infos(masks_all_ones)],
            target_out=target_out,
            loss_type=config.output_loss_type,
            device=device,
        )
        total_loss += config.out_recon_coeff * out_recon_loss
        loss_terms["output_recon"] = out_recon_loss.item()

    # Embedding reconstruction loss
    if config.embedding_recon_coeff is not None:
        stochastic_masks_list = calc_stochastic_masks(
            causal_importances=causal_importances,
            n_mask_samples=config.n_mask_samples,
            sampling=config.sampling,
        )
        embedding_recon_loss = calc_embedding_recon_loss(
            model=model,
            batch=batch,
            masks=[stochastic_masks.component_masks for stochastic_masks in stochastic_masks_list],
            unembed=config.is_embed_unembed_recon,
            device=device,
        )
        total_loss += config.embedding_recon_coeff * embedding_recon_loss
        loss_terms["embedding_recon"] = embedding_recon_loss.item()

    loss_terms["total"] = total_loss.item()

    return total_loss, loss_terms
