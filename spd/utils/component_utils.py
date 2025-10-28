from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.configs import SamplingType
from spd.models.components import (
    ComponentsMaskInfo,
    WeightDeltaAndMask,
    make_mask_infos,
)


def rand_perm(
    shape: tuple[int, ...],
    dim: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Int[Tensor, "... k"]:
    """Create a LongTensor of shape `shape` containing random permutations along dimension `dim`.
    For example, if shape is (2, 3) and dim is 1, the returned tensor will be a 2x3 tensor with
    each row having a random permutation of [0, 1, 2].

    Args:
        shape: Shape of the tensor to create
        dim: Dimension along which to make the permutations
        device: Device to create the tensor on
        generator: Generator to use for the random values

    Returns:
        LongTensor of shape `shape` with randomly ordered permutation along dimension `dim`.
    """

    noise = torch.rand(shape, device=device, generator=generator)
    # turn values into ranks via double argsort trick. (for example: [0.8, 0.2, 0.3] -> [2, 0, 1])
    return noise.argsort(dim=dim).argsort(dim=dim)


def sample_uniform_k_subset_routing_masks(
    mask_shape: tuple[int, ...],
    module_names: list[str],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, Bool[Tensor, "..."]]:
    """Creates routing masks for each module such that the number of modules routed to for each
    position is independent and uniformly sampled from [1, len(module_names)]

    Achieves this by:
    - for each position, k is independent and uniformly sampled from [1, len(module_names)]
    - for each position, a k-sized random subset of modules are routed to

    Args:
        mask_shape: Shape of the routing masks, likely (batch,) or (batch, seq_len)
        module_names: List of module names to route to

    Returns:
        Dict mapping module names to routing masks of shape `mask_shape`.
    """
    k_modules_to_route: Int[Tensor, " ..."] = torch.randint(
        low=1,
        high=len(module_names) + 1,
        size=mask_shape,
        device=device,
        generator=generator,
    )

    perms: Int[Tensor, "k_modules ..."] = rand_perm(
        shape=(len(module_names), *mask_shape),
        dim=0,
        device=device,
        generator=generator,
    )

    return {mod: perms[i] < k_modules_to_route for i, mod in enumerate(module_names)}


RoutingType = Literal["uniform_k-stochastic", "all"]
"""How to choose which (batch,) or (batch, seq_len) positions to route to components or target.

uniform_k-stochastic:
    for each position, sample k from [1, n_modules], then route to components for k out of
    `n_modules` modules
all:
    use components for all positions
"""


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    routing: RoutingType,
) -> dict[str, ComponentsMaskInfo]:
    ci_sample = next(iter(causal_importances.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device
    dtype = ci_sample.dtype

    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        match component_mask_sampling:
            case "binomial":
                stochastic_source = torch.randint(0, 2, ci.shape, device=device).float()
            case "continuous":
                stochastic_source = torch.rand_like(ci)
        component_masks[layer] = ci + (1 - ci) * stochastic_source

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_deltas is not None:
        weight_deltas_and_masks = {}
        for layer in causal_importances:
            weight_deltas_and_masks[layer] = (
                weight_deltas[layer],
                torch.rand(leading_dims, device=device, dtype=dtype),
            )

    match routing:
        case "all":
            routing_masks = "all"
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=leading_dims, module_names=list(causal_importances.keys()), device=device
            )

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
