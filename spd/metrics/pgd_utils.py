from typing import Literal, Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import (
    RoutingType,
    calc_routing_masks,
)
from spd.utils.general_utils import calc_sum_recon_loss_lm, zip_dicts


class PGDObjective(Protocol):
    def __call__(
        self,
        *,
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor: ...


def get_pgd_init_tensor(
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


def optimize_adversarial_stochastic_masks(
    *,
    pgd_config: PGDConfig,
    objective_fn: PGDObjective,
    example_shape: tuple[int, ...],
    device: torch.device | str,
    use_delta_component: bool,
    layers: list[str],
) -> tuple[
    dict[str, Float[Tensor, "... C"]],
    dict[str, Float[Tensor, "..."]] | None,
]:
    """Optimize adversarial stochastic sources in [0,1] via PGD to maximize stochastic losses.

    Returns (rand_tensors, weight_delta_rand_masks) where:
    - rand_tensors: per-layer tensors shaped like corresponding causal_importances
    - weight_delta_rand_masks: per-layer tensors with leading dims matching inputs (or None)
    """

    # batch dims is either (B,) or (B, S)
    *batch_dims, C = example_shape

    def repeat_masks(
        ci: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "..."]] | None]:
        match pgd_config.mask_scope:
            case "unique_per_datapoint":
                # no-op
                return ci, weight_delta_mask
            case "shared_across_batch":
                cm = {}
                for module_name in ci:
                    cm[module_name] = ci[module_name].repeat(*batch_dims, 1)

                wdm = None
                if weight_delta_mask is not None:
                    wdm = {}
                    for module_name in weight_delta_mask:
                        wdm[module_name] = weight_delta_mask[module_name].repeat(*batch_dims)
                return cm, wdm

    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            ci_mask_shape = torch.Size(batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(batch_dims)
        case "shared_across_batch":
            # replace all batch dims with 1 so we broadcast the same mask across all datapoints
            ci_mask_shape = torch.Size(1 for _ in batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(1 for _ in batch_dims)

    component_mask: dict[str, Float[Tensor, "... C"]] = {}
    for layer in layers:
        mask = get_pgd_init_tensor(pgd_config.init, ci_mask_shape, device)
        mask.requires_grad_(True)
        component_mask[layer] = mask

    weight_delta_mask: dict[str, Float[Tensor, " ..."]] | None = None
    if use_delta_component:
        weight_delta_mask = {}
        for layer in layers:
            wd_init = get_pgd_init_tensor(pgd_config.init, weight_delta_mask_shape, device)
            wd_init.requires_grad_(True)
            weight_delta_mask[layer] = wd_init

    # PGD ascent
    for _ in range(int(pgd_config.n_steps)):
        adv_vars = list(component_mask.values())
        if weight_delta_mask is not None:
            adv_vars.extend(list(weight_delta_mask.values()))

        assert all(v.requires_grad for v in adv_vars)

        with torch.enable_grad():
            component_mask_, weight_delta_mask_ = repeat_masks(component_mask, weight_delta_mask)
            obj = objective_fn(
                component_mask=component_mask_,
                weight_delta_mask=weight_delta_mask_,
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

    for module_name in component_mask:
        component_mask[module_name].detach_()
    if weight_delta_mask is not None:
        for module_name in weight_delta_mask:
            weight_delta_mask[module_name].detach_()

    return repeat_masks(component_mask, weight_delta_mask)


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

    def objective_fn(
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor:
        match weight_deltas, weight_delta_mask:
            case None, None:
                weight_deltas_and_masks = None
            case dict(), dict():
                weight_deltas_and_masks = zip_dicts(weight_deltas, weight_delta_mask)
            case _:
                raise ValueError(
                    "weight_deltas and weight_delta_mask must exist or not exist together"
                )

        mask_infos = make_mask_infos(
            component_masks=component_mask,
            weight_deltas_and_masks=weight_deltas_and_masks,
            routing_masks=routing_masks,
        )

        out = model(batch, mask_infos=mask_infos)
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
        n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
        return total_loss / n_examples

    adversarial_component_masks, adversarial_weight_delta_masks = (
        optimize_adversarial_stochastic_masks(
            pgd_config=pgd_config,
            objective_fn=objective_fn,
            layers=list(ci.keys()),
            example_shape=next(iter(ci.values())).shape,
            device=batch.device,
            use_delta_component=weight_deltas is not None,
        )
    )

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
        component_masks=adversarial_component_masks,
        weight_deltas_and_masks=adversarial_weight_deltas_and_masks,
        routing_masks=routing_masks,
    )

    out = model(batch, mask_infos=sampled_mask_infos)
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
    return loss, n_examples
