from typing import Literal, Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import (
    RoutingType,
    WeightDeltaSamplingData,
    calc_stochastic_component_mask_info,
)
from spd.utils.general_utils import calc_sum_recon_loss_lm, zip_dicts


class PGDObjective(Protocol):
    def __call__(
        self,
        *,
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor: ...


PGDInitStrategy = Literal["random", "ones", "zeroes"]


def get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.full(shape, 1.0, device=device)
        case "zeroes":
            return torch.full(shape, 0.0, device=device)


MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


def create_forward_pass_as_function_of_masks(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
):
    def obj(
        component_mask: dict[str, Float[Tensor, "... C"]],
        weight_delta_mask: dict[str, Float[Tensor, "..."]] | None,
    ) -> Tensor:
        # Stochastic reconstruction loss (all components at once)
        weight_deltas_and_mask_values: (
            tuple[dict[str, Float[Tensor, "d_out d_in"]], WeightDeltaSamplingData] | None
        ) = None
        assert (weight_deltas is None) == (weight_delta_mask is None)
        if weight_delta_mask is not None:
            assert weight_deltas is not None
            weight_deltas_and_mask_values = (weight_deltas, ("given", weight_delta_mask))

        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=("given", component_mask),
            weight_deltas_and_mask_sampling=weight_deltas_and_mask_values,
            routing=routing,
        )
        out = model(batch, mask_infos=mask_infos)
        loss_type = output_loss_type
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
        n_examples = out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()
        return total_loss / n_examples

    return obj


def optimize_adversarial_stochastic_masks(
    *,
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    objective_fn: PGDObjective,
    example_shape: tuple[int, ...],
    device: torch.device | str,
    use_delta_component: bool,
    mask_scope: MaskScope,
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

    match mask_scope:
        case "shared_across_batch":
            ci_mask_shape = example_shape
            weight_delta_mask_shape = example_shape[:-1]
        case "unique_per_datapoint":
            # ones for leading dims apart from the C dim - use the same C mask for all datapoints
            ci_mask_shape = torch.Size(1 for _ in example_shape[:-1]) + example_shape[-1:]
            # ones for all dims - use the same weight delta mask for all datapoints
            weight_delta_mask_shape = torch.Size(1 for _ in example_shape[:-1])

    component_mask: dict[str, Float[Tensor, "... C"]] = {}
    for layer in layers:
        mask = get_pgd_init_tensor(init, ci_mask_shape, device)
        mask.requires_grad_(True)
        component_mask[layer] = mask

    weight_delta_mask: dict[str, Float[Tensor, " ..."]] | None = None
    if use_delta_component:
        weight_delta_mask = {}
        for layer in layers:
            wd_init = get_pgd_init_tensor(init, weight_delta_mask_shape, device)
            wd_init.requires_grad_(True)
            weight_delta_mask[layer] = wd_init

    # PGD ascent
    for _ in range(int(n_steps)):
        for v in component_mask.values():
            assert v.grad is not None
        if weight_delta_mask is not None:
            for v in weight_delta_mask.values():
                assert v.grad is not None

        adv_vars = list(component_mask.values())
        if weight_delta_mask is not None:
            adv_vars.extend(list(weight_delta_mask.values()))

        obj = objective_fn(
            component_mask=component_mask,
            weight_delta_mask=weight_delta_mask,
        )

        grads = torch.autograd.grad(
            obj,
            adv_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # REMOVEME
        assert all(isinstance(g, Tensor) for g in grads)
        print(f"PASSED GRAD CHECK --- REMOVE NOW {__file__}")
        # ========

        with torch.no_grad():
            for v, g in zip(adv_vars, grads, strict=True):
                v.add_(step_size * g.sign())
                v.clamp_(0.0, 1.0)
                v.detach_()

    return component_mask, weight_delta_mask


def pgd_masked_recon_loss_update(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    mask_scope: MaskScope,
) -> tuple[Float[Tensor, ""], int]:
    objective_fn = create_forward_pass_as_function_of_masks(
        model=model,
        batch=batch,
        target_out=target_out,
        output_loss_type=output_loss_type,
        routing=routing,
        ci=ci,
        weight_deltas=weight_deltas,
    )

    component_masks, weight_delta_masks = optimize_adversarial_stochastic_masks(
        init=init,
        step_size=step_size,
        n_steps=n_steps,
        objective_fn=objective_fn,
        layers=list(ci.keys()),
        example_shape=next(iter(ci.values())).shape,
        device=batch.device,
        use_delta_component=weight_deltas is not None,
        mask_scope=mask_scope,
    )

    assert (weight_delta_masks is None) == (weight_deltas is None)
    weight_deltas_and_masks = None
    if weight_delta_masks is not None:
        assert weight_deltas is not None
        weight_deltas_and_masks = zip_dicts(weight_deltas, weight_delta_masks)

    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks="all",
    )

    out = model(batch, mask_infos=mask_infos)
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
    return loss, n_examples
