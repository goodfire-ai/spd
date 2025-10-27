import uuid
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import RoutingType, calc_routing_masks
from spd.utils.distributed_utils import (
    all_reduce,
    call_on_rank0_then_broadcast,
    gather_all_tensors,
    get_rank,
    sync_across_processes,
)
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
        rank = get_rank()
        print(f"{rank} - making mask infos")
        mask_infos = make_mask_infos(
            component_masks=_interpolate_component_mask(ci, component_sample_points),
            weight_deltas_and_masks=weight_deltas_and_masks,
            routing_masks=routing_masks,
        )
        print(f"{rank} - calling model")
        out = model(batch, mask_infos=mask_infos)
        print(f"{rank} - calling calc_sum_recon_loss_lm")
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
        n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
        print(f"{rank} - done")
        return total_loss, n_examples

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
        source = call_on_rank0_then_broadcast(
            _get_pgd_init_tensor,
            pgd_config.init,
            ci_mask_shape,
            batch.device,
        ).requires_grad_(True)
        assert_same_across_ranks(source)
        adversarial_component_sample_points[layer] = source

    adversarial_weight_delta_masks: dict[str, Float[Tensor, " ..."]] | None = None
    if weight_deltas is not None:
        adversarial_weight_delta_masks = {}
        for layer in ci:
            source = call_on_rank0_then_broadcast(
                _get_pgd_init_tensor,
                pgd_config.init,
                weight_delta_mask_shape,
                batch.device,
            ).requires_grad_(True)
            assert_same_across_ranks(source)
            adversarial_weight_delta_masks[layer] = source

    # PGD ascent
    adv_sources = list(adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        adv_sources.extend(list(adversarial_weight_delta_masks.values()))

    for i in range(int(pgd_config.n_steps)):
        assert all(v.requires_grad for v in adv_sources)
        assert all(v.grad is None for v in adv_sources)

        with torch.enable_grad():
            total_loss, _ = objective_fn(  # n_examples doesn't matter bc we're doing sign ascent
                *maybe_repeat_to_batch_shape(
                    adversarial_component_sample_points, adversarial_weight_delta_masks
                )
            )
            grads = torch.autograd.grad(total_loss, adv_sources)

        reduced_grads = [all_reduce(grad.clone(), op=ReduceOp.SUM) for grad in grads]
        for grad in reduced_grads:
            assert_same_across_ranks(grad)

        with torch.no_grad():
            for source, grad in zip(adv_sources, reduced_grads, strict=True):
                source.add_(pgd_config.step_size * grad.sign())
                source.clamp_(0.0, 1.0)
                assert_same_across_ranks(source)

    for source in adv_sources:
        source.detach_()

    adversarial_component_sample_points, adversarial_weight_delta_masks = (
        maybe_repeat_to_batch_shape(
            adversarial_component_sample_points, adversarial_weight_delta_masks
        )
    )

    for source in adversarial_component_sample_points.values():
        assert_same_across_ranks(source)
    if adversarial_weight_delta_masks is not None:
        for source in adversarial_weight_delta_masks.values():
            assert_same_across_ranks(source)

    assert all(not v.requires_grad for v in adversarial_component_sample_points.values())
    if adversarial_weight_delta_masks is not None:
        assert not any(v.requires_grad for v in adversarial_weight_delta_masks.values())

    total_loss, n_examples = objective_fn(
        adversarial_component_sample_points, adversarial_weight_delta_masks
    )

    # no need to all-reduce total_loss or n_examples bc consumers handle this

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


def assert_same_across_ranks(tensor: Tensor) -> None:
    tensors = gather_all_tensors(tensor.clone())
    for t in tensors:
        assert t.shape == tensor.shape
        assert torch.all(t == tensor)
    sync_across_processes()
