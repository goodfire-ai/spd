from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    make_mask_infos,
)
from spd.utils.component_utils import (
    calc_stochastic_component_mask_info,
    sample_uniform_k_subset_routing_masks,
)
from spd.utils.general_utils import calc_kl_divergence_lm


@dataclass(frozen=True)
class MaskedForwardCache:
    outputs: list[Float[Tensor, "... d_model_out"]]
    hidden_acts_by_layer: dict[str, list[Float[Tensor, "..."]]]


def run_masked_forward(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    mask_infos_list: list[dict[str, ComponentsMaskInfo]],
    hidden_module_names: Iterable[str] | None = None,
) -> MaskedForwardCache:
    output_cache: list[Tensor] = []
    hidden_acts_cache: dict[str, list[Tensor]] = {m: [] for m in (hidden_module_names or [])}
    for mask_infos in mask_infos_list:
        if hidden_module_names:
            # Run with both components and input cache to get hidden activations from the model WITH components
            output, hidden_acts = model(
                batch,
                mode="components_input_cache",
                mask_infos=mask_infos,
                module_names=list(hidden_module_names),
            )
            for name in hidden_acts_cache:
                hidden_acts_cache[name].append(hidden_acts[name])
        else:
            output = model(batch, mode="components", mask_infos=mask_infos)
        output_cache.append(output)
    return MaskedForwardCache(outputs=output_cache, hidden_acts_by_layer=hidden_acts_cache)


def output_recon_loss_from_cache(
    cache: MaskedForwardCache,
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"],
) -> Float[Tensor, ""]:
    if loss_type == "mse":
        return torch.stack([((o - target_out) ** 2).mean() for o in cache.outputs]).mean()
    return torch.stack(
        [calc_kl_divergence_lm(pred=o, target=target_out) for o in cache.outputs]
    ).mean()


def hidden_recon_losses_from_cache(
    cache: MaskedForwardCache,
    target_hidden: dict[str, Tensor],
) -> dict[str, Float[Tensor, ""]]:
    return {
        layer_name: torch.stack(
            [((a - target_hidden[layer_name]) ** 2).mean() for a in layer_acts]
        ).mean()
        for layer_name, layer_acts in cache.hidden_acts_by_layer.items()
    }


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

    assert len(model.components) == 1, "Only one embedding component is supported"
    components = next(iter(model.components.values()))
    assert isinstance(components, EmbeddingComponents)

    # --- original embedding output --------------------------------------------------------- #
    # Get the target model's embedding layer
    target_model = model.target_model
    embedding_layer = None
    for _, module in target_model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_layer = module
            break
    assert embedding_layer is not None, "No embedding layer found in target model"
    target_out: Float[Tensor, "... d_emb"] = embedding_layer(batch)

    # --- masked embedding output ----------------------------------------------------------- #
    loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        assert len(mask_info) == 1, "Only one embedding component is supported"
        mask = next(iter(mask_info.values()))
        masked_out: Float[Tensor, "... d_emb"] = components(batch, mask=mask)

        if unembed:
            assert hasattr(model.target_model, "lm_head"), "Only supports unembedding named lm_head"
            assert isinstance(model.target_model.lm_head, nn.Module)
            target_out_unembed = model.target_model.lm_head(target_out)
            masked_out_unembed = model.target_model.lm_head(masked_out)
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
    output_recon_loss_type: Literal["mse", "kl"],
) -> Float[Tensor, ""]:
    """Calculate the recon loss when applying all (masked) component layers at once.

    This function takes the mean loss over all masks in mask_infos_list.

    Args:
        model: The component model
        batch: Input batch
        mask_infos_list: Mask infos for each stochastic source (there are config.n_mask_samples
            stochastic sources).
        target_out: Target model output
        output_recon_loss_type: Type of loss to calculate for output reconstruction
        device: Device to run computations on

    Returns:
        The recon loss
    """
    # Do a forward pass with all components
    assert output_recon_loss_type in ["mse", "kl"], f"Invalid loss type: {output_recon_loss_type}"

    total_loss = torch.tensor(0.0, device=batch.device)
    for mask_infos in mask_infos_list:
        out = model(batch, mode="components", mask_infos=mask_infos)
        if output_recon_loss_type == "mse":
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
    # device parameter kept for backward compatibility
    _ = device
    return model.calc_weight_deltas()


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
    target_hidden: dict[str, Tensor],
    device: str,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
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
        target_hidden: Dictionary of target hidden activations for each layer
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
            output_recon_loss_type=config.output_recon_loss_type,
        )
        total_loss += config.ci_recon_coeff * ci_recon_loss
        loss_terms["ci_recon"] = ci_recon_loss.item()

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        stoch_mask_infos_list = [
            calc_stochastic_component_mask_info(
                causal_importances=causal_importances,
                sampling=config.sampling,
                routing="all",
                weight_deltas=weight_deltas if config.use_delta_component else None,
            )
            for _ in range(config.n_mask_samples)
        ]
        cache = run_masked_forward(
            model=model,
            batch=batch,
            mask_infos_list=stoch_mask_infos_list,
            hidden_module_names=target_hidden.keys()
            if config.hidden_act_recon_coeff is not None
            else None,
        )
        stochastic_recon_loss = output_recon_loss_from_cache(
            cache, target_out, config.output_recon_loss_type
        )
        if config.hidden_act_recon_coeff is not None:
            hidden_losses = hidden_recon_losses_from_cache(cache, target_hidden)
            for layer_name, layer_loss in hidden_losses.items():
                total_loss += config.hidden_act_recon_coeff * layer_loss
                loss_terms[f"hidden_act_recon/{layer_name}"] = layer_loss.item()
        total_loss += config.stochastic_recon_coeff * stochastic_recon_loss
        loss_terms["stochastic_recon"] = stochastic_recon_loss.item()

    # CI reconstruction layerwise loss
    if config.ci_recon_layerwise_coeff is not None:
        ci_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            mask_infos_list=[make_mask_infos(causal_importances, weight_deltas_and_masks=None)],
            target_out=target_out,
            loss_type=config.output_recon_loss_type,
            device=device,
        )
        total_loss += config.ci_recon_layerwise_coeff * ci_recon_layerwise_loss
        loss_terms["ci_recon_layerwise"] = ci_recon_layerwise_loss.item()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
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
            loss_type=config.output_recon_loss_type,
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
            output_recon_loss_type=config.output_recon_loss_type,
        )
        total_loss += config.ci_masked_recon_subset_coeff * ci_recon_subset_loss
        loss_terms["ci_recon_subset"] = ci_recon_subset_loss.item()

    # Stochastic reconstruction subset loss
    if config.stochastic_recon_subset_coeff is not None:
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
            output_recon_loss_type=config.output_recon_loss_type,
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
