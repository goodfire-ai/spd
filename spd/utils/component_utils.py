from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor


@dataclass
class LayerMasks:
    """Mask information for a given layer."""

    routing_mask: Bool[Tensor, " ..."] | bool
    """whether to use the target or component layer for a given example (either (b,) or (b, s))"""
    component_mask: Float[Tensor, "... C"]
    """The sub-component mask to use when using the component layer."""
    weight_delta_mask: Float[Tensor, "..."]
    """The mask to use if using the weight delta component. This doesn't have a final C dim: it's
    scalar for each example."""


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sampling: Literal["continuous", "binomial"],
) -> list[dict[str, LayerMasks]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic sources to calculate. I.e. the number of times
            we sample a mask for each layer.
        sampling: The sampling method to use.

    Return:
        StochasticMasks object for each stochastic source.
    """

    stochastic_masks_list: list[dict[str, LayerMasks]] = []

    for _ in range(n_mask_samples):
        stochastic_masks: dict[str, LayerMasks] = {}

        # static Ps across layers
        ci_sample: Float[Tensor, "... C"] = next(iter(causal_importances.values()))
        p_vals = torch.rand(ci_sample.shape[:-1], device=ci_sample.device, dtype=ci_sample.dtype)

        for layer, ci in causal_importances.items():
            routing_mask = torch.rand_like(p_vals) > p_vals

            if sampling == "binomial":
                rand_tensor = torch.randint(0, 2, ci.shape, device=ci.device).float()
            else:
                rand_tensor = torch.rand_like(ci)
            component_mask = ci + (1 - ci) * rand_tensor

            weight_delta_mask = torch.rand(ci.shape[:-1], device=ci.device, dtype=ci.dtype)

            stochastic_masks[layer] = LayerMasks(
                routing_mask,
                component_mask,
                weight_delta_mask,
            )

        stochastic_masks_list.append(stochastic_masks)

    return stochastic_masks_list


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
