from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import RoutingType, calc_routing_masks
from spd.utils.distributed_utils import all_reduce, call_on_rank0_then_broadcast
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

    *batch_dims, C = next(iter(ci.values())).shape
    batch_device = next(iter(ci.values())).device

    def init_and_sync_adversarial_tensor(shape: torch.Size) -> Float[Tensor, "..."]:
        return (
            call_on_rank0_then_broadcast(
                lambda: _get_pgd_init_tensor(pgd_config.init, shape, "cpu")
            )
            .to(batch_device)
            .requires_grad_(True)
        )

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
        adversarial_component_sample_points[layer] = init_and_sync_adversarial_tensor(ci_mask_shape)

    adversarial_weight_delta_masks: dict[str, Float[Tensor, " ..."]] | None = None
    if weight_deltas is not None:
        adversarial_weight_delta_masks = {}
        for layer in ci:
            adversarial_weight_delta_masks[layer] = init_and_sync_adversarial_tensor(
                weight_delta_mask_shape
            )

    # PGD ascent
    adv_vars = list(adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        adv_vars.extend(list(adversarial_weight_delta_masks.values()))

    for _ in range(int(pgd_config.n_steps)):
        assert all(v.requires_grad for v in adv_vars)
        assert all(v.grad is None for v in adv_vars)

        with torch.enable_grad():
            total_loss, _ = objective_fn( # n_examples doesn't matter bc we're doing sign ascent
                *maybe_repeat_to_batch_shape(
                    adversarial_component_sample_points, adversarial_weight_delta_masks
                )
            )
            grads = torch.autograd.grad(total_loss, adv_vars)

        reduced_grads = [all_reduce(grad.clone(), op=ReduceOp.SUM) for grad in grads]

        with torch.no_grad():
            for v, g in zip(adv_vars, reduced_grads, strict=True):
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

    total_loss, n_examples = objective_fn(
        adversarial_component_sample_points, adversarial_weight_delta_masks
    )

    total_loss = all_reduce(total_loss.clone())
    n_examples_tensor = all_reduce(total_loss.new_tensor(float(n_examples)))
    n_examples = int(n_examples_tensor.item())

    return total_loss, n_examples


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
