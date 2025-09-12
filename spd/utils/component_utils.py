from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sampling: Literal["continuous", "binomial"],
) -> tuple[list[dict[str, Float[Tensor, "... C"]]], list[dict[str, Float[Tensor, "..."]]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
    Return:
        - List of stochastic masks for each layer.
        - List of weight delta masks for each layer.
    """

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []
    # weight_delta_mask have the same leading dims as causal_importances but no final dim
    weight_delta_masks: list[dict[str, Float[Tensor, ...]]] = []

    for _ in range(n_mask_samples):
        stochastic_mask: dict[str, Float[Tensor, "... C"]] = {}
        weight_delta_mask: dict[str, Float[Tensor, ...]] = {}
        for layer, ci in causal_importances.items():
            if sampling == "binomial":
                rand_tensor = torch.randint(0, 2, ci.shape, device=ci.device).float()
            else:
                rand_tensor = torch.rand_like(ci)
            stochastic_mask[layer] = ci + (1 - ci) * rand_tensor

            weight_delta_mask[layer] = torch.rand(ci.shape[:-1], device=ci.device, dtype=ci.dtype)

        stochastic_masks.append(stochastic_mask)
        weight_delta_masks.append(weight_delta_mask)

    return stochastic_masks, weight_delta_masks


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
