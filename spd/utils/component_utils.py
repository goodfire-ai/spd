from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sampling: Literal["continuous", "binomial"],
) -> tuple[list[dict[str, Float[Tensor, "... C"]]], list[dict[str, Float[Tensor, "..."]]]]:
    """Calculate n_mask_samples stochastic masks and corresponding r values.

    Stochastic masks are computed as:
        ci + (1 - ci) * base - r
    where base is either binomial(0/1) or uniform(0,1) depending on ``sampling``,
    and ``r`` is a dictionary with the same keys as ``causal_importances`` but
    lacks the trailing C dimension (so it broadcasts across components).

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
        sampling: "continuous" uses uniform noise, "binomial" uses 0/1 draws.
    Return:
        Tuple of:
            - list of n_mask_samples dictionaries of stochastic masks per layer (shape ... C)
            - list of n_mask_samples dictionaries of r per layer (shape ...)
    """

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []
    rs: list[dict[str, Float[Tensor, ...]]] = []

    for _ in range(n_mask_samples):
        r_dict: dict[str, Float[Tensor, ...]] = {
            layer: torch.rand_like(ci[..., 0]) for layer, ci in causal_importances.items()
        }

        if sampling == "binomial":
            mask_dict = {
                layer: ci
                + (1 - ci) * torch.randint(0, 2, ci.shape, device=ci.device).float()
                - r_dict[layer][..., None]
                for layer, ci in causal_importances.items()
            }
        else:
            mask_dict = {
                layer: ci + (1 - ci) * torch.rand_like(ci) - r_dict[layer][..., None]
                for layer, ci in causal_importances.items()
            }

        stochastic_masks.append(mask_dict)
        rs.append(r_dict)

    return stochastic_masks, rs


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
