from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.models.components import ComponentMaskInfo, WeightDeltaAndMask, make_mask_infos


def sample_stochastic_mask(
    causal_importances: Float[Tensor, "... C"],
    sampling: Literal["continuous", "binomial"],
) -> Float[Tensor, "... C"]:
    if sampling == "binomial":
        rand_tensor = torch.randint(
            0, 2, causal_importances.shape, device=causal_importances.device
        ).float()
    else:
        rand_tensor = torch.rand_like(causal_importances)
    return causal_importances + (1 - causal_importances) * rand_tensor


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    sampling: Literal["continuous", "binomial"],
    weight_deltas: dict[str, Tensor] | None,
    routing: Literal["variable-p", "all"],
) -> dict[str, ComponentMaskInfo]:
    routing_masks: dict[str, Bool[Tensor, " ..."] | bool] = {}
    # static Ps across layers
    ci_sample: Float[Tensor, "... C"] = next(iter(causal_importances.values()))
    p_vals = torch.rand(ci_sample.shape[:-1], device=ci_sample.device, dtype=ci_sample.dtype)
    for layer in causal_importances:
        routing_masks[layer] = torch.rand_like(p_vals) > p_vals if routing == "variable-p" else True

    component_masks: dict[str, Float[Tensor, "... C"] | bool] = {}
    for layer, ci in causal_importances.items():
        component_masks[layer] = sample_stochastic_mask(ci, sampling)

    if weight_deltas is not None:
        weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = {}
        for layer, ci in causal_importances.items():
            mask = torch.rand(ci.shape[:-1], device=ci.device, dtype=ci.dtype)
            weight_deltas_and_masks[layer] = (weight_deltas[layer], mask)
    else:
        weight_deltas_and_masks = None

    return make_mask_infos(component_masks, routing_masks, weight_deltas_and_masks)


def calc_stochastic_component_mask_infos(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    sampling: Literal["continuous", "binomial"],
    weight_deltas: dict[str, Tensor] | None,
    routing: Literal["variable-p", "all"],
    n_mask_samples: int,
) -> list[dict[str, ComponentMaskInfo]]:
    return [
        calc_stochastic_component_mask_info(causal_importances, sampling, weight_deltas, routing)
        for _ in range(n_mask_samples)
    ]


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
