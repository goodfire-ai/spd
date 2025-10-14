from functools import partial
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import RoutingMasks, make_mask_infos
from spd.utils.component_utils import RoutingType, calc_routing_masks
from spd.utils.general_utils import calc_sum_recon_loss_lm, zip_dicts


def pgd_masked_recon_loss_update(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Central implementation of PGD masked reconstruction loss.

    Optimizes adversarial stochastic masks and optionally weight deltas for the given objective function.
    """

    # Shared routing mask within the whole function
    routing_masks = calc_routing_masks(
        routing,
        leading_dims=next(iter(ci.values())).shape[:-1],
        module_names=list(ci.keys()),
        device=batch.device,
    )

    example_shape = next(iter(ci.values())).shape
    device = batch.device
    use_delta_component = weight_deltas is not None

    # batch dims is either (B,) or (B, S)
    *batch_dims, C = example_shape

    def maybe_repeat_to_batch_shape(
        component_sample_points: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "..."]] | None]:
        match pgd_config.mask_scope:
            case "unique_per_datapoint":
                # no-op
                return component_sample_points, weight_delta_mask
            case "shared_across_batch":
                csp_out = {}
                correct_sample_points_shape = torch.Size(1 for _ in batch_dims) + (C,)
                for module in component_sample_points:
                    assert component_sample_points[module].shape == correct_sample_points_shape
                    csp_out[module] = component_sample_points[module].repeat(*batch_dims, 1)

                wdm_out = None
                if weight_delta_mask is not None:
                    wdm_out = {}
                    correct_wdm_shape = torch.Size(1 for _ in batch_dims)
                    for module in weight_delta_mask:
                        assert weight_delta_mask[module].shape == correct_wdm_shape
                        wdm_out[module] = weight_delta_mask[module].repeat(*batch_dims)

                return csp_out, wdm_out

    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            ci_mask_shape = torch.Size(batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(batch_dims)
        case "shared_across_batch":
            # replace all batch dims with 1 so we broadcast the same mask across all datapoints
            ci_mask_shape = torch.Size(1 for _ in batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(1 for _ in batch_dims)

    adversarial_component_sample_points: dict[str, Float[Tensor, "... C"]] = {}
    for layer in ci:
        sample_point = _get_pgd_init_tensor(pgd_config.init, ci_mask_shape, device)
        sample_point.requires_grad_(True)
        adversarial_component_sample_points[layer] = sample_point

    adversarial_weight_delta_masks: dict[str, Float[Tensor, " ..."]] | None = None
    if use_delta_component:
        adversarial_weight_delta_masks = {}
        for layer in ci:
            wd_init = _get_pgd_init_tensor(pgd_config.init, weight_delta_mask_shape, device)
            wd_init.requires_grad_(True)
            adversarial_weight_delta_masks[layer] = wd_init

    objective_as_fn_of_adversarial_args = partial(
        objective_fn,
        model=model,
        batch=batch,
        target_out=target_out,
        output_loss_type=output_loss_type,
        ci=ci,
        routing_masks=routing_masks,
        weight_deltas=weight_deltas,
    )

    # PGD ascent
    adv_vars = list(adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        adv_vars.extend(list(adversarial_weight_delta_masks.values()))

    for _ in range(int(pgd_config.n_steps)):
        assert all(v.requires_grad for v in adv_vars)
        assert all(v.grad is None for v in adv_vars)

        with torch.enable_grad():
            adversarial_component_sample_points_, adversarial_weight_delta_masks_ = (
                maybe_repeat_to_batch_shape(
                    adversarial_component_sample_points, adversarial_weight_delta_masks
                )
            )

            obj = objective_as_fn_of_adversarial_args(
                component_sample_points=adversarial_component_sample_points_,
                weight_delta_masks=adversarial_weight_delta_masks_,
            )

            grads = torch.autograd.grad(
                obj,
                adv_vars,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        with torch.no_grad():
            for v, g in zip(adv_vars, grads, strict=True):
                v.add_(pgd_config.step_size * g.sign())
                v.clamp_(0.0, 1.0)
                g.zero_()

    for var in adv_vars:
        var.detach_()

    match weight_deltas, adversarial_weight_delta_masks:
        case None, None:
            adversarial_weight_deltas_and_masks = None
        case dict(), dict():
            adversarial_weight_deltas_and_masks = zip_dicts(
                weight_deltas, adversarial_weight_delta_masks
            )
        case _:
            raise ValueError(
                "weight_deltas and adversarial_weight_delta_masks must exist or not exist together"
            )

    sampled_mask_infos = make_mask_infos(
        component_masks=_interpolate_component_mask(ci, adversarial_component_sample_points),
        weight_deltas_and_masks=adversarial_weight_deltas_and_masks,
        routing_masks=routing_masks,
    )

    out = model(batch, mask_infos=sampled_mask_infos)
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
    return loss, n_examples


def _get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "... C"]],
    component_sample_points: dict[str, Float[Tensor, "... C"]],
) -> dict[str, Float[Tensor, "... C"]]:
    component_mask = {}
    for module_name in ci:
        component_mask[module_name] = (
            ci[module_name] + (1 - ci[module_name]) * component_sample_points[module_name]
        )
    return component_mask


def objective_fn(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    ci: dict[str, Float[Tensor, "... C"]],
    routing_masks: RoutingMasks,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    # The things we're actually optimizing:
    component_sample_points: dict[str, Float[Tensor, "... C"]],
    weight_delta_masks: dict[str, Float[Tensor, "..."]] | None,
) -> Tensor:
    match weight_deltas, weight_delta_masks:
        case None, None:
            weight_deltas_and_masks = None
        case dict(), dict():
            weight_deltas_and_masks = zip_dicts(weight_deltas, weight_delta_masks)
        case _:
            raise ValueError("weight_deltas and weight_delta_mask must exist or not exist together")

    mask_infos = make_mask_infos(
        component_masks=_interpolate_component_mask(ci, component_sample_points),
        weight_deltas_and_masks=weight_deltas_and_masks,  # we don't interpolate for the weight delta
        routing_masks=routing_masks,
    )

    out = model(batch, mask_infos=mask_infos)
    total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
    return total_loss / n_examples
