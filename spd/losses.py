import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsOrModule

# def calc_importance_minimality_loss(
#     ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float, eps: float = 1e-12
# ) -> Float[Tensor, ""]:
#     """Calculate the importance minimality loss on the upper leaky relu causal importances.

#     Args:
#         ci_upper_leaky: Dictionary of causal importances upper leaky relu for each layer.
#         pnorm: The pnorm to use for the importance minimality loss. Must be positive.
#         eps: The epsilon to add to the causal importances to avoid division by zero when computing
#             the gradients for pnorm < 1.

#     Returns:
#         The importance minimality loss on the upper leaky relu causal importances.
#     """
#     total_loss = torch.zeros_like(next(iter(ci_upper_leaky.values())))

#     for layer_ci_upper_leaky in ci_upper_leaky.values():
#         # Note, the paper uses an absolute value but our layer_ci_upper_leaky is already > 0
#         total_loss = total_loss + (layer_ci_upper_leaky + eps) ** pnorm

#     # Sum over the C dimension and mean over the other dimensions
#     return total_loss.sum(dim=-1).mean()


# def calc_masked_recon_layerwise_loss(
#     model: ComponentModel,
#     batch: Int[Tensor, "..."],
#     mask_infos_list: list[dict[str, ComponentsMaskInfo]],
#     target_out: Float[Tensor, "... d_model_out"],
#     loss_type: Literal["mse", "kl"],
#     device: str,
# ) -> Float[Tensor, ""]:
#     """Calculate the recon loss when augmenting the model one (masked) component layer at a time.

#     This function takes the mean loss over all masks in mask_infos_list.

#     Args:
#         model: The component model
#         batch: Input batch
#         mask_infos_list: Mask infos for each stochastic source (there are config.n_mask_samples
#             stochastic sources).
#         target_out: Target model output
#         loss_type: Type of loss to calculate
#         device: Device to run computations on

#     Returns:
#         The recon loss
#     """
#     assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
#     total_loss = torch.tensor(0.0, device=device)
#     for mask_infos in mask_infos_list:
#         for module_name, mask_info in mask_infos.items():
#             modified_out = model(batch, mode="components", mask_infos={module_name: mask_info})
#             if loss_type == "mse":
#                 loss = ((modified_out - target_out) ** 2).mean()
#             else:
#                 loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
#             total_loss += loss
#     n_modified_components = len(mask_infos_list[0])
#     n_stochastic_sources = len(mask_infos_list)
#     return total_loss / (n_modified_components * n_stochastic_sources)


# def calc_masked_recon_loss(
#     model: ComponentModel,
#     batch: Float[Tensor, "... d_in"],
#     mask_infos_list: list[dict[str, ComponentsMaskInfo]],
#     target_out: Float[Tensor, "... d_model_out"],
#     loss_type: Literal["mse", "kl"],
#     device: str,
# ) -> Float[Tensor, ""]:
#     """Calculate the recon loss when applying all (masked) component layers at once.

#     This function takes the mean loss over all masks in mask_infos_list.

#     Args:
#         model: The component model
#         batch: Input batch
#         mask_infos_list: Mask infos for each stochastic source (there are config.n_mask_samples
#             stochastic sources).
#         target_out: Target model output
#         loss_type: Type of loss to calculate
#         device: Device to run computations on

#     Returns:
#         The recon loss
#     """
#     # Do a forward pass with all components
#     assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"

#     total_loss = torch.tensor(0.0, device=device)
#     for mask_infos in mask_infos_list:
#         out = model(batch, mode="components", mask_infos=mask_infos)
#         if loss_type == "mse":
#             loss = ((out - target_out) ** 2).mean()
#         else:
#             loss = calc_kl_divergence_lm(pred=out, target=target_out)
#         total_loss += loss

#     return total_loss / len(mask_infos_list)


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


# def calc_faithfulness_loss(
#     weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
#     device: str | torch.device,
# ) -> Float[Tensor, ""]:
#     """Calculate the MSE loss between component weights (V@U) and target weights.

#     We sum over all layers and normalize by the number of parameters in the model (this includes any
#     inserted identity matrices).
#     """

#     n_params = sum(param.numel() for param in weight_deltas.values())
#     mse = torch.tensor(0.0, device=device)
#     for param in weight_deltas.values():
#         mse += ((param) ** 2).sum()
#     # Normalize by the number of parameters in the model (including any inserted identity matrices)
#     mse = mse / n_params
#     return mse


def compute_total_loss(
    loss_metric_configs: list[MetricConfig],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    ci: dict[str, Float[Tensor, "batch C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    device: str,
    current_p: float | None = None,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict). terms_dict contains raw per-term values (no coeffs) and a weighted total.
    """
    total = torch.tensor(0.0, device=device)
    terms: dict[str, float] = {}

    for loss_metric_config in loss_metric_configs:
        loss_metric_cls: type[Metric] = METRICS[loss_metric_config.classname]
        loss_metric = loss_metric_cls(
            model, config, **loss_metric_config.extra_init_kwargs, sync_on_compute=False
        ).to(device)
        loss = loss_metric(
            batch=batch,
            target_out=target_out,
            ci=ci,
            ci_upper_leaky=ci_upper_leaky,
            weight_deltas=weight_deltas,
        )
        total = total + loss_metric_config.coeff * loss
        terms[loss_metric_config.classname] = loss.item()

    # if config.faithfulness_coeff is not None:
    #     val = calc_faithfulness_loss(weight_deltas, device)
    #     total = total + config.faithfulness_coeff * val
    #     terms["faithfulness"] = float(val.item())

    # if config.ci_recon_coeff is not None:
    #     ci_recon_loss = calc_masked_recon_loss(
    #         model=model,
    #         batch=batch,
    #         mask_infos_list=[make_mask_infos(ci)],
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #         device=device,
    #     )
    #     total = total + config.ci_recon_coeff * ci_recon_loss
    #     terms["ci_recon"] = ci_recon_loss.item()

    # if config.stochastic_recon_coeff is not None:
    #     stochastic_masks_list = calc_stochastic_masks(
    #         causal_importances=ci,
    #         n_mask_samples=config.n_mask_samples,
    #         sampling=config.sampling,
    #     )
    #     mask_infos_list = []
    #     for stochastic_masks in stochastic_masks_list:
    #         deltas_and_masks = (
    #             {
    #                 key: (weight_deltas[key], stochastic_masks.weight_delta_masks[key])
    #                 for key in weight_deltas
    #             }
    #             if config.use_delta_component
    #             else None
    #         )
    #         mask_infos_list.append(
    #             make_mask_infos(
    #                 masks=stochastic_masks.component_masks, weight_deltas_and_masks=deltas_and_masks
    #             )
    #         )

    #     stochastic_recon_loss = calc_masked_recon_loss(
    #         model=model,
    #         batch=batch,
    #         mask_infos_list=mask_infos_list,
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #         device=device,
    #     )

    #     total = total + config.stochastic_recon_coeff * stochastic_recon_loss
    #     terms["stochastic_recon"] = stochastic_recon_loss.item()

    # if config.ci_recon_layerwise_coeff is not None:
    #     ci_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
    #         model=model,
    #         batch=batch,
    #         mask_infos_list=[make_mask_infos(ci)],
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #         device=device,
    #     )
    #     total = total + config.ci_recon_layerwise_coeff * ci_recon_layerwise_loss
    #     terms["ci_recon_layerwise"] = ci_recon_layerwise_loss.item()

    # if config.stochastic_recon_layerwise_coeff is not None:
    #     stochastic_masks_list = calc_stochastic_masks(
    #         causal_importances=ci,
    #         n_mask_samples=config.n_mask_samples,
    #         sampling=config.sampling,
    #     )
    #     mask_infos_list_2: list[dict[str, ComponentsMaskInfo]] = []
    #     for stochastic_masks in stochastic_masks_list:
    #         weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = (
    #             {
    #                 key: (weight_deltas[key], stochastic_masks.weight_delta_masks[key])
    #                 for key in weight_deltas
    #             }
    #             if config.use_delta_component
    #             else None
    #         )
    #         mask_infos_list_2.append(
    #             make_mask_infos(
    #                 masks=stochastic_masks.component_masks,
    #                 weight_deltas_and_masks=weight_deltas_and_masks,
    #             )
    #         )

    #     stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
    #         model=model,
    #         batch=batch,
    #         mask_infos_list=mask_infos_list_2,
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #         device=device,
    #     )
    #     total = total + config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
    #     terms["stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # pnorm_value = current_p if current_p is not None else config.pnorm
    # val = calc_importance_minimality_loss(ci_upper_leaky=ci_upper_leaky, pnorm=pnorm_value)
    # total = total + config.importance_minimality_coeff * val
    # terms["importance_minimality"] = float(val.item())

    terms["total"] = total.item()

    return total, terms
