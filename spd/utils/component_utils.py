import torch
from jaxtyping import Float
from torch import Tensor


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []

    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {layer: ci + (1 - ci) * torch.rand_like(ci) for layer, ci in causal_importances.items()}
        )

    return stochastic_masks


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
