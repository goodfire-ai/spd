from collections.abc import Callable
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import RoutingType, calc_routing_masks
from spd.utils.general_utils import calc_sum_recon_loss_lm, zip_dicts

ObjFn = Callable[
    [dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "..."]] | None],
    tuple[Float[Tensor, ""], int],
]


def _run_pgd_optimization(
    *,
    objective_fn: ObjFn,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    batch_device: torch.device | str,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Run PGD optimization for any objective function."""

    *batch_dims, C = next(iter(ci.values())).shape

    def maybe_repeat_to_batch_shape(
        component_sample_points: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "..."]] | None]:
        return (
            {k: v.expand(*batch_dims, -1) for k, v in component_sample_points.items()},
            {k: v.expand(*batch_dims) for k, v in weight_delta_mask.items()}
            if weight_delta_mask is not None
            else None,
        )

    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            ci_mask_shape = torch.Size(batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(batch_dims)
        case "shared_across_batch":
            ci_mask_shape = torch.Size(1 for _ in batch_dims) + (C,)
            weight_delta_mask_shape = torch.Size(1 for _ in batch_dims)

    adversarial_component_sample_points: dict[str, Float[Tensor, "... C"]] = {}
    for layer in ci:
        sample_point = _get_pgd_init_tensor(pgd_config.init, ci_mask_shape, batch_device)
        sample_point.requires_grad_(True)
        adversarial_component_sample_points[layer] = sample_point

    adversarial_weight_delta_masks: dict[str, Float[Tensor, " ..."]] | None = None
    if weight_deltas is not None:
        adversarial_weight_delta_masks = {}
        for layer in ci:
            wd_init = _get_pgd_init_tensor(pgd_config.init, weight_delta_mask_shape, batch_device)
            wd_init.requires_grad_(True)
            adversarial_weight_delta_masks[layer] = wd_init

    # PGD ascent
    adv_vars = list(adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        adv_vars.extend(list(adversarial_weight_delta_masks.values()))

    for _ in range(int(pgd_config.n_steps)):
        assert all(v.requires_grad for v in adv_vars)
        assert all(v.grad is None for v in adv_vars)

        with torch.enable_grad():
            loss, n_examples = objective_fn(
                *maybe_repeat_to_batch_shape(
                    adversarial_component_sample_points, adversarial_weight_delta_masks
                )
            )
            obj = loss / n_examples

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

    for var in adv_vars:
        var.detach_()

    adversarial_component_sample_points, adversarial_weight_delta_masks = (
        maybe_repeat_to_batch_shape(
            adversarial_component_sample_points, adversarial_weight_delta_masks
        )
    )
    assert all(not v.requires_grad for v in adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        assert not any(v.requires_grad for v in adversarial_weight_delta_masks.values())

    return objective_fn(adversarial_component_sample_points, adversarial_weight_delta_masks)


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

    routing_masks = calc_routing_masks(
        routing,
        leading_dims=next(iter(ci.values())).shape[:-1],
        module_names=list(ci.keys()),
        device=batch.device,
    )

    def objective_fn(
        component_sample_points: dict[str, Float[Tensor, "... C"]],
        weight_delta_masks: dict[str, Float[Tensor, "..."]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        match weight_deltas, weight_delta_masks:
            case None, None:
                weight_deltas_and_masks = None
            case dict(), dict():
                weight_deltas_and_masks = zip_dicts(weight_deltas, weight_delta_masks)
            case _:
                raise ValueError(
                    "weight_deltas and weight_delta_mask must exist or not exist together"
                )
        mask_infos = make_mask_infos(
            component_masks=_interpolate_component_mask(ci, component_sample_points),
            weight_deltas_and_masks=weight_deltas_and_masks,
            routing_masks=routing_masks,
        )
        out = model(batch, mask_infos=mask_infos)
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
        n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
        return total_loss, n_examples

    return _run_pgd_optimization(
        objective_fn=objective_fn,
        ci=ci,
        weight_deltas=weight_deltas,
        batch_device=batch.device,
        pgd_config=pgd_config,
    )


def pgd_masked_hidden_acts_recon_loss_update(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    post_target_module_path: str,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Central implementation of PGD masked reconstruction loss.

    Optimizes adversarial stochastic masks and optionally weight deltas for the given objective function.
    """

    target_post_acts = model(
        batch,
        cache_type="output",
        cache_points=[post_target_module_path],
    ).cache[post_target_module_path]

    def objective_fn(
        component_sample_points: dict[str, Float[Tensor, "... C"]],
        weight_delta_masks: dict[str, Float[Tensor, "..."]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        match weight_deltas, weight_delta_masks:
            case None, None:
                weight_deltas_and_masks = None
            case dict(), dict():
                weight_deltas_and_masks = zip_dicts(weight_deltas, weight_delta_masks)
            case _:
                raise ValueError(
                    "weight_deltas and weight_delta_mask must exist or not exist together"
                )
        mask_infos = make_mask_infos(
            component_masks=_interpolate_component_mask(ci, component_sample_points),
            weight_deltas_and_masks=weight_deltas_and_masks,
        )

        stoch_post_acts = model(
            batch,
            cache_type="output",
            mask_infos=mask_infos,
            cache_points=[post_target_module_path],
        ).cache[post_target_module_path]

        mse = torch.nn.functional.mse_loss(target_post_acts, stoch_post_acts, reduction="sum")
        assert not batch.is_floating_point()
        n_examples = batch.shape.numel()
        return mse, n_examples

    return _run_pgd_optimization(
        objective_fn=objective_fn,
        ci=ci,
        weight_deltas=weight_deltas,
        batch_device=batch.device,
        pgd_config=pgd_config,
    )


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
        assert torch.all(component_sample_points[module_name] <= 1.0)

        scaled_noise_to_add = (1 - ci[module_name]) * component_sample_points[module_name]
        assert torch.all(scaled_noise_to_add >= 0.0)
        assert torch.all(scaled_noise_to_add <= 1.0 - ci[module_name])

        component_mask[module_name] = ci[module_name] + scaled_noise_to_add
        assert torch.all(component_mask[module_name] >= ci[module_name])
        assert torch.all(component_mask[module_name] <= 1.0)
    return component_mask
