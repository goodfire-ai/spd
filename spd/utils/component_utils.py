from jaxtyping import Float
from torch import Tensor

from spd.utils.distributed_utils import get_distributed_rand_like


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    step: int,
    hash_prefix: str,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Handles producing masks that are the same regardless of the number of distributed processes.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
        step: The current training step. Used to seed the random number generator.
        hash_prefix: A string used to seed the random number generator.
    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []

    for sample_idx in range(n_mask_samples):
        mask_dict = {}
        for layer, ci in causal_importances.items():
            hash_key = f"{hash_prefix}-{step}-{sample_idx}-{layer}"
            rand_vals = get_distributed_rand_like(ci.shape, hash_key, device=ci.device)
            mask_dict[layer] = ci + (1 - ci) * rand_vals

        stochastic_masks.append(mask_dict)

    return stochastic_masks


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
