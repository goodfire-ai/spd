import hashlib

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import get_rank, is_initialized


def _get_distributed_rand_like(
    shape: tuple[int, ...], step: int, sample_idx: int, layer: str
) -> Tensor:
    """Get a tensor of shape `shape` which matches an indexed tensor on the current rank.

    This function simulates the following process:
    1. Generate a full random tensor of shape `shape * world_size` on CPU with a custom generator
    2. Index the tensor to get the portion of the tensor on the current rank
    3. Move the indexed portion to the GPU

    It does this by iterating through random values until the rng counter matches what is needed
    for the current rank.

    Args:
        shape: The shape of the tensor to get.
        step: The current training step. Used to seed the random number generator.
        sample_idx: The current sample index. Used to seed the random number generator.
        layer: The layer name. Used to seed the random number generator.
    """
    rank = get_rank() if is_initialized() else 0

    # Assert that shape has 3 dimensions (batch, seq_len, C). In future we'd want to support other
    # shapes
    assert len(shape) == 3, "Shape must have 3 dimensions (batch, seq_len, C)"
    local_batch_size, seq_len, C = shape

    generator = torch.Generator(device="cpu")
    seed = int(hashlib.md5(f"{step}-{sample_idx}-{layer}".encode()).hexdigest(), 16) % (2**32)
    generator.manual_seed(seed)

    elements_per_sample = seq_len * C
    total_elements_to_skip = rank * local_batch_size * elements_per_sample

    skip_chunk_size = 100_000
    remaining = total_elements_to_skip
    while remaining > 0:
        chunk = min(remaining, skip_chunk_size)
        torch.rand(chunk, generator=generator)
        remaining -= chunk

    # Generate this rank's data
    return torch.rand(*shape, device="cpu", generator=generator)


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    step: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.
        step: The current training step. Used to seed the random number generator.
    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """

    stochastic_masks: list[dict[str, Float[Tensor, "... C"]]] = []

    for sample_idx in range(n_mask_samples):
        mask_dict = {}
        for layer, ci in causal_importances.items():
            rand_vals = _get_distributed_rand_like(ci.shape, step, sample_idx, layer).to(ci.device)
            mask_dict[layer] = ci + (1 - ci) * rand_vals

        stochastic_masks.append(mask_dict)

    return stochastic_masks


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
