from typing import Literal, Protocol, cast

import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel


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
    dtype: torch.dtype,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device, dtype=dtype)
        case "ones":
            return torch.full(shape, 1.0, device=device, dtype=dtype)
        case "zeroes":
            return torch.full(shape, 0.0, device=device, dtype=dtype)


def optimize_adversarial_stochastic_masks(
    *,
    model: ComponentModel,
    init: PGDInitStrategy,
    # device: str,
    step_size: float,
    n_steps: int,
    objective: PGDObjective,
    causal_importances: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> tuple[
    dict[str, Float[Tensor, "... C"]],
    dict[str, Float[Tensor, "..."]] | None,
]:
    """Optimize adversarial stochastic sources in [0,1] via PGD to maximize stochastic losses.

    Returns (rand_tensors, weight_delta_rand_masks) where:
    - rand_tensors: per-layer tensors shaped like corresponding causal_importances
    - weight_delta_rand_masks: per-layer tensors with leading dims matching inputs (or None)
    """
    # Initialize adversarial variables
    ci_sample = next(iter(causal_importances.values()))

    ci_dims = ci_sample.shape
    device = ci_sample.device
    leading_dims = ci_dims[:-1]

    component_mask: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        mask = get_pgd_init_tensor(init, ci_dims, device, ci.dtype)
        mask.requires_grad_(True)
        component_mask[layer] = mask

    weight_delta_mask: dict[str, Float[Tensor, " ..."]] | None = None
    if weight_deltas is not None:
        weight_delta_mask = {}
        for layer, ci in causal_importances.items():
            wd_init = get_pgd_init_tensor(init, leading_dims, device, ci.dtype)
            wd_init.requires_grad_(True)
            weight_delta_mask[layer] = wd_init

    # Temporarily disable grads for model params to avoid building graphs w.r.t. parameters
    # and reduce memory usage during PGD inner optimization. Restore afterwards.
    prev_train_mode = model.training
    prev_requires_grad: list[bool] = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    # Use eval mode to avoid stochastic layers affecting adversarial objective
    model.eval()

    obj_value = objective(
        component_mask=component_mask,
        weight_delta_mask=weight_delta_mask,
    )

    try:
        # PGD ascent
        for _ in range(int(n_steps)):
            # Zero any existing grads on rand tensors
            for v in component_mask.values():
                if v.grad is not None:
                    v.grad = None
            if weight_delta_mask is not None:
                for v in weight_delta_mask.values():
                    if v.grad is not None:
                        v.grad = None

            adv_vars = list(component_mask.values()) + (
                list(weight_delta_mask.values()) if weight_delta_mask is not None else []
            )
            raw_grads = torch.autograd.grad(
                obj_value,
                adv_vars,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            grads = cast(tuple[Tensor | None, ...], raw_grads)

            with torch.no_grad():
                # Update all adversarial variables in the same order they were passed to autograd
                for v, g in zip(adv_vars, grads, strict=True):
                    if g is not None:
                        v.add_(step_size * g.sign())
                    v.clamp_(0.0, 1.0)
    finally:
        # Restore model state
        for p, req in zip(model.parameters(), prev_requires_grad, strict=True):
            p.requires_grad_(req)
        model.train(prev_train_mode)

    # Detach to avoid tracking grads in the outer loss backward
    with torch.no_grad():
        for layer in list(component_mask.keys()):
            component_mask[layer].detach_().clamp_(0.0, 1.0)
        if weight_delta_mask is not None:
            for layer in list(weight_delta_mask.keys()):
                weight_delta_mask[layer].detach_().clamp_(0.0, 1.0)

    return component_mask, weight_delta_mask
