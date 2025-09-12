from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sampling: Literal["continuous", "binomial"],
) -> tuple[list[dict[str, Float[Tensor, "... C"]]], dict[str, Float[Tensor, "..."]]]:
    """Calculate n_mask_samples stochastic masks and the corresponding r dictionary.

    Stochastic masks are calculated per layer as:
        mask = ci + (1 - ci) * X - r
    where:
        - ci is the causal importance tensor with a trailing C dimension
        - X is either Bernoulli({0,1}) (when sampling == "binomial") or Uniform(0,1)
          sampled with the same shape as ci
        - r is a tensor in [0,1) with the same leading dimensions as ci but without the C dim

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
        sampling: "continuous" to sample X ~ Uniform(0,1), "binomial" for X ~ Bernoulli({0,1}).

    Returns:
        (stochastic_masks, r):
            - stochastic_masks: list of length n_mask_samples containing per-layer masks
            - r: dictionary of tensors like causal_importances but without the C dimension
    """

    # r has the same leading dims as ci but without the trailing C dim
    r: dict[str, Float[Tensor, ...]] = {
        layer: torch.rand(ci.shape[:-1], device=ci.device, dtype=ci.dtype)
        for layer, ci in causal_importances.items()
    }

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []

    for _ in range(n_mask_samples):
        sample: dict[str, Float[Tensor, "... C"]] = {}
        for layer, ci in causal_importances.items():
            if sampling == "binomial":
                x_sample = torch.randint(0, 2, ci.shape, device=ci.device).float()
            else:
                x_sample = torch.rand_like(ci)
            # Broadcast r without the C dim across the last (C) dimension
            r_broadcast = r[layer].unsqueeze(-1)
            sample[layer] = ci + (1 - ci) * x_sample - r_broadcast
        stochastic_masks.append(sample)

    return stochastic_masks, r


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
