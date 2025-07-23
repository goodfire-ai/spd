from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent
from spd.utils.component_utils import calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm


def calc_embedding_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    component: EmbeddingComponent,
    masks: list[dict[str, Float[Tensor, "... C"]]],
    embed_module_name: str,
    unembed: bool = False,
) -> Float[Tensor, ""]:
    """
    recon loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponent``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the masked embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the masked embedding output
    and the target embedding output is used as the loss.
    """

    # --- original embedding output --------------------------------------------------------- #
    orig_module = model.model.get_submodule(embed_module_name)
    assert isinstance(orig_module, nn.Embedding), (
        f"Module {embed_module_name} expected to be nn.Embedding, got {type(orig_module)}"
    )
    target_out: Float[Tensor, "... d_emb"] = orig_module(batch)

    # --- masked embedding output ----------------------------------------------------------- #
    loss = torch.tensor(0.0, device=component.V.device)
    for mask_info in masks:
        component.mask = mask_info[embed_module_name]

        masked_out: Float[Tensor, "... d_emb"] = component(batch)
        component.mask = None

        if unembed:
            assert hasattr(model.model, "lm_head"), "Only supports unembedding named lm_head"
            assert isinstance(model.model.lm_head, nn.Module)
            target_out_unembed = model.model.lm_head(target_out)
            masked_out_unembed = model.model.lm_head(masked_out)
            loss += calc_kl_divergence_lm(pred=masked_out_unembed, target=target_out_unembed)
        else:
            loss += ((masked_out - target_out) ** 2).sum(dim=-1).mean()

    loss /= len(masks)

    return loss


def calc_schatten_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    components: dict[str, LinearComponent | EmbeddingComponent],
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
    device: str,
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"] = "kl",
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            modified_out = model.forward_with_components(
                batch,
                components={component_name: component},
                masks={component_name: mask_info[component_name]},
            )
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            else:
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))


def calc_masked_recon_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_mdoel_out"],
    loss_type: Literal["mse", "kl"] = "mse",
) -> Float[Tensor, ""]:
    """Calculate the MSE over all masks."""
    # Do a forward pass with all components
    out = model.forward_with_components(batch, components=components, masks=masks)
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
    if loss_type == "mse":
        loss = ((out - target_out) ** 2).mean()
    else:
        loss = calc_kl_divergence_lm(pred=out, target=target_out)

    return loss


def _calc_tensors_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    faithfulness_loss = torch.tensor(0.0, device=device)
    for name in params1:
        faithfulness_loss = faithfulness_loss + ((params2[name] - params1[name]) ** 2).sum()
    return faithfulness_loss / n_params


def calc_faithfulness_loss(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (V@U + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)
        assert isinstance(submodule, nn.Linear | nn.Embedding)
        target_params[comp_name] = submodule.weight
        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss = _calc_tensors_mse(
        params1=component_params,
        params2=target_params,
        n_params=n_params,
        device=device,
    )
    return faithfulness_loss


def calc_ce_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "..."]],
    unmasked_component_logits: Float[Tensor, "..."],
    masked_component_logits: Float[Tensor, "..."],
    target_logits: Float[Tensor, "..."],
) -> dict[str, float]:
    """Calculate cross-entropy losses for various masking scenarios.

    Args:
        model: The component model
        batch: Input batch
        components: Dictionary of components
        masks: Dictionary of masks for components
        unmasked_component_logits: Logits from unmasked components
        masked_component_logits: Logits from masked components
        target_logits: Target model logits

    Returns:
        Dictionary containing CE losses for different scenarios
    """
    ce_losses: dict[str, float] = {}

    # Flatten logits and batch for CE calculation
    flat_all_component_logits = einops.rearrange(
        unmasked_component_logits, "... vocab -> (...) vocab"
    )
    flat_masked_component_logits = einops.rearrange(
        masked_component_logits, "... vocab -> (...) vocab"
    )
    flat_batch = batch.flatten()

    # CE vs true labels
    unmasked_ce_loss = F.cross_entropy(input=flat_all_component_logits[:-1], target=flat_batch[1:])
    masked_ce_loss = F.cross_entropy(input=flat_masked_component_logits[:-1], target=flat_batch[1:])

    flat_target_logits = einops.rearrange(target_logits, "... vocab -> (...) vocab")
    target_ce_loss = F.cross_entropy(input=flat_target_logits[:-1], target=flat_batch[1:])

    # CE when every component is fully masked (all-zero masks)
    zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
    zero_masked_component_logits = model.forward_with_components(
        batch, components=components, masks=zero_masks
    )
    flat_zero_masked_component_logits = einops.rearrange(
        zero_masked_component_logits, "... vocab -> (...) vocab"
    )
    zero_masked_ce_loss = F.cross_entropy(
        input=flat_zero_masked_component_logits[:-1], target=flat_batch[1:]
    )

    ce_losses["misc/unmasked_ce_loss_vs_labels"] = unmasked_ce_loss.item()
    ce_losses["misc/masked_ce_loss_vs_labels"] = masked_ce_loss.item()
    ce_losses["misc/target_ce_loss_vs_labels"] = target_ce_loss.item()
    ce_losses["misc/zero_masked_ce_loss_vs_labels"] = zero_masked_ce_loss.item()

    return ce_losses


def calc_cosine_similarity_loss(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
) -> Float[Tensor, ""]:
    """Calculate the cosine similarity loss between subcomponents and target model parameters.
    This loss discourages negative cosine similarity between subcomponents and target model parameters.
    The loss is the mean of max(0, -cosine_similarity), so it only penalizes negative similarities.
    When cosine similarity is 0 or positive, the loss is 0.
    Args:
        components: Dictionary mapping component names to components
        target_model: The target model containing the original parameters
    Returns:
        The loss value (mean of max(0, -cosine_similarity)), which is always non-negative and lower is better
    """
    total_negative_cosine_loss = torch.tensor(0.0, device=next(iter(components.values())).V.device)
    total_subcomponents = 0

    for comp_name, component in components.items():
        # Get target weight matrix
        target_submodule = target_model.get_submodule(comp_name)
        assert isinstance(target_submodule, nn.Linear | nn.Embedding)
        target_weight = target_submodule.weight

        # Get V and U matrices from component
        V = component.V  # (d_in, C) for Linear or (vocab_size, C) for Embedding
        U = component.U  # (C, d_out) for Linear or (C, embedding_dim) for Embedding

        # Calculate individual subcomponent weights: V[:, c] @ U[c, :] for each c
        # For LinearComponent: weight = U^T @ V^T, so subcomp = V[:, c] @ U[c, :]^T = V[:, c:c+1] @ U[c:c+1, :]
        # For EmbeddingComponent: weight = V @ U, so subcomp = V[:, c:c+1] @ U[c:c+1, :]

        if isinstance(component, EmbeddingComponent):
            # EmbeddingComponent: V @ U -> (vocab_size, embedding_dim)
            subcomp_weights = einops.einsum(
                V, U, "vocab_size C, C embedding_dim -> C vocab_size embedding_dim"
            )
        else:
            # LinearComponent: U^T @ V^T -> (d_out, d_in)
            subcomp_weights = einops.einsum(V, U, "d_in C, C d_out -> C d_out d_in")

        # Flatten for cosine similarity calculation
        subcomp_flat = subcomp_weights.flatten(start_dim=1)  # (C, d_out * d_in)
        target_flat = target_weight.flatten()  # (d_out * d_in)

        # Calculate cosine similarity for all subcomponents at once
        cos_sims = F.cosine_similarity(subcomp_flat, target_flat, dim=1)  # (C,)

        # Calculate loss: max(0, -cosine_similarity) - only penalizes negative similarities
        negative_cosine_loss = torch.clamp(-cos_sims, min=0.0)  # (C,)

        # Sum up negative cosine losses
        total_negative_cosine_loss += negative_cosine_loss.sum()
        total_subcomponents += component.C

    # Return mean of negative cosine losses (lower is better, consistent with other losses)
    mean_negative_cosine_loss = total_negative_cosine_loss / total_subcomponents
    return mean_negative_cosine_loss


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    components: dict[str, LinearComponent | EmbeddingComponent],
    causal_importances: dict[str, Float[Tensor, "batch C"]],
    causal_importances_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    device: str,
    n_params: int,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Calculate all losses and return total loss and individual loss terms.

    Args:
        model: The component model
        batch: Input batch
        config: Configuration object with loss coefficients
        components: Dictionary of component modules
        causal_importances: Causal importance masks
        causal_importances_upper_leaky: Upper leaky causal importances for regularization
        target_out: Target model output
        device: Device to run computations on
        n_params: Total number of parameters in the model

    Returns:
        Tuple of (total_loss, loss_terms_dict)
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_terms: dict[str, float] = {}

    # Faithfulness loss
    if config.faithfulness_coeff is not None:
        faithfulness_loss = calc_faithfulness_loss(
            components=components, target_model=model.model, n_params=n_params, device=device
        )
        total_loss += config.faithfulness_coeff * faithfulness_loss
        loss_terms["loss/faithfulness"] = faithfulness_loss.item()

    # Reconstruction loss
    if config.recon_coeff is not None:
        recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            components=components,
            masks=causal_importances,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_coeff * recon_loss
        loss_terms["loss/recon"] = recon_loss.item()

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        stochastic_recon_loss = torch.tensor(0.0, device=target_out.device)
        for i in range(len(stochastic_masks)):
            stochastic_recon_loss += calc_masked_recon_loss(
                model=model,
                batch=batch,
                components=components,
                masks=stochastic_masks[i],
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
        stochastic_recon_loss = stochastic_recon_loss / len(stochastic_masks)
        total_loss += config.stochastic_recon_coeff * stochastic_recon_loss
        loss_terms["loss/stochastic_recon"] = stochastic_recon_loss.item()

    # Reconstruction layerwise loss
    if config.recon_layerwise_coeff is not None:
        recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=[causal_importances],
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_layerwise_coeff * recon_layerwise_loss
        loss_terms["loss/recon_layerwise"] = recon_layerwise_loss.item()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
        layerwise_stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=layerwise_stochastic_masks,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
        loss_terms["loss/stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # Importance minimality loss
    importance_minimality_loss = calc_importance_minimality_loss(
        ci_upper_leaky=causal_importances_upper_leaky, pnorm=config.pnorm
    )
    total_loss += config.importance_minimality_coeff * importance_minimality_loss
    loss_terms["loss/importance_minimality"] = importance_minimality_loss.item()

    # Schatten loss
    if config.schatten_coeff is not None:
        schatten_loss = calc_schatten_loss(
            ci_upper_leaky=causal_importances_upper_leaky,
            pnorm=config.pnorm,
            components=components,
            device=device,
        )
        total_loss += config.schatten_coeff * schatten_loss
        loss_terms["loss/schatten"] = schatten_loss.item()

    # Output reconstruction loss
    if config.out_recon_coeff is not None:
        masks_all_ones = {k: torch.ones_like(v) for k, v in causal_importances.items()}
        out_recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            components=components,
            masks=masks_all_ones,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.out_recon_coeff * out_recon_loss
        loss_terms["loss/output_recon"] = out_recon_loss.item()

    # Embedding reconstruction loss
    if config.embedding_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        assert len(components) == 1, "Only one embedding component is supported"
        component = list(components.values())[0]
        assert isinstance(component, EmbeddingComponent)
        embedding_recon_loss = calc_embedding_recon_loss(
            model=model,
            batch=batch,
            component=component,
            masks=stochastic_masks,
            embed_module_name=next(iter(components.keys())),
            unembed=config.is_embed_unembed_recon,
        )
        total_loss += config.embedding_recon_coeff * embedding_recon_loss
        loss_terms["loss/embedding_recon"] = embedding_recon_loss.item()

        # Cosine similarity loss
    if config.cosine_similarity_coeff is not None:
        cosine_similarity_loss = calc_cosine_similarity_loss(
            components=components, target_model=model.model
        )
        total_loss += config.cosine_similarity_coeff * cosine_similarity_loss
        loss_terms["loss/cosine_similarity"] = cosine_similarity_loss.item()

    return total_loss, loss_terms
