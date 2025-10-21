from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.models.components import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos


def _sample_stochastic_mask(
    causal_importances: Float[Tensor, "... C"],
    sampling: Literal["continuous", "binomial"],
) -> Float[Tensor, "... C"]:
    match sampling:
        case "binomial":
            rand_tensor = torch.randint(
                0, 2, causal_importances.shape, device=causal_importances.device
            ).float()
        case "continuous":
            rand_tensor = torch.rand_like(causal_importances)

    return causal_importances + (1 - causal_importances) * rand_tensor


RoutingType = Literal["uniform_k-stochastic", "all"]
"""How to choose which (batch,) or (batch, seq_len) positions to route to components or target.

uniform_k-stochastic:
    for each position, sample k from [1, n_modules], then route to components for k out of
    `n_modules` modules
all:
    use components for all positions
"""


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


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    sampling: Literal["continuous", "binomial"],
    routing: RoutingType,
    weight_deltas: dict[str, Tensor] | None,
) -> dict[str, ComponentsMaskInfo]:
    ci_sample = next(iter(causal_importances.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device
    dtype = ci_sample.dtype

    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        component_masks[layer] = _sample_stochastic_mask(ci, sampling)

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None
    if weight_deltas is not None:
        weight_deltas_and_masks = {
            layer: (weight_deltas[layer], torch.rand(leading_dims, device=device, dtype=dtype))
            for layer in causal_importances
        }
    else:
        weight_deltas_and_masks = None

    match routing:
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                leading_dims,
                list(causal_importances.keys()),
                device,
            )
        case "all":
            routing_masks = None

    return make_mask_infos(
        component_masks=component_masks,
        routing_masks=routing_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
    )


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
