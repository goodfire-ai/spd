import random
from typing import Any, Literal, Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.consts import ClusterCoactivationShaped, MergePair

MergePairSamplerKey = Literal["range", "mcmc"]


class MergePairSamplerConfigurable(Protocol):
    def __call__(
        self,
        costs: ClusterCoactivationShaped,
        **kwargs: Any,
    ) -> MergePair: ...


class MergePairSampler(Protocol):
    def __call__(
        self,
        costs: ClusterCoactivationShaped,
    ) -> MergePair: ...


def range_sampler(
    costs: ClusterCoactivationShaped,
    threshold: float = 0.05,
    **kwargs: Any,
) -> MergePair:
    """Sample a merge pair using threshold-based range selection.

    Considers all pairs with costs below a threshold defined as a fraction
    of the range of non-diagonal costs, then randomly selects one.

    Args:
        costs: Cost matrix for all possible merges (may contain NaN for invalid pairs)
        k_groups: Number of current groups
        threshold: Fraction of cost range to consider (0=min only, 1=all pairs)

    Returns:
        Tuple of (group_i, group_j) indices to merge
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    # Mask out NaN entries and diagonal
    valid_mask: Bool[Tensor, "k_groups k_groups"] = ~torch.isnan(costs)
    diag_mask: Bool[Tensor, "k_groups k_groups"] = ~torch.eye(
        k_groups, dtype=torch.bool, device=costs.device
    )
    valid_mask = valid_mask & diag_mask

    # Get valid costs
    valid_costs: Float[Tensor, " n_valid"] = costs[valid_mask]

    if valid_costs.numel() == 0:
        raise ValueError("All costs are NaN, cannot sample merge pair")

    # Find the range of valid costs
    min_cost: float = float(valid_costs.min().item())
    max_cost: float = float(valid_costs.max().item())

    # Calculate threshold cost
    max_considered_cost: float = (max_cost - min_cost) * threshold + min_cost

    # Find all valid pairs below threshold
    within_range: Bool[Tensor, "k_groups k_groups"] = (costs <= max_considered_cost) & valid_mask

    # Get indices of candidate pairs
    considered_idxs: Int[Tensor, "n_considered 2"] = torch.stack(
        torch.where(within_range), dim=1
    )

    if considered_idxs.shape[0] == 0:
        raise ValueError("No valid pairs within threshold range")

    # Randomly select one of the considered pairs
    selected_idx: int = random.randint(0, considered_idxs.shape[0] - 1)
    pair_tuple: tuple[int, int] = tuple(considered_idxs[selected_idx].tolist())  # type: ignore[assignment]
    return MergePair(pair_tuple)


def mcmc_sampler(
    costs: ClusterCoactivationShaped,
    temperature: float = 1.0,
    **kwargs: Any,
) -> MergePair:
    """Sample a merge pair using MCMC with probability proportional to exp(-cost/temperature).

    Args:
        costs: Cost matrix for all possible merges (may contain NaN for invalid pairs)
        k_groups: Number of current groups
        temperature: Temperature parameter for softmax (higher = more uniform sampling)

    Returns:
        Tuple of (group_i, group_j) indices to merge
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    # Create mask for valid pairs (non-diagonal and non-NaN)
    valid_mask: Bool[Tensor, "k_groups k_groups"] = ~torch.eye(
        k_groups, dtype=torch.bool, device=costs.device
    )
    valid_mask = valid_mask & ~torch.isnan(costs)

    # Check if we have any valid pairs
    if not valid_mask.any():
        raise ValueError("All costs are NaN, cannot sample merge pair")

    # Compute probabilities: exp(-cost/temperature)
    # Use stable softmax computation to avoid overflow
    costs_masked: ClusterCoactivationShaped = costs.clone()
    costs_masked[~valid_mask] = float("inf")  # Set invalid entries to inf so exp gives 0

    # Subtract min for numerical stability
    min_cost: float = float(costs_masked[valid_mask].min())
    probs: ClusterCoactivationShaped = (
        torch.exp((min_cost - costs_masked) / temperature) * valid_mask
    )  # Zero out invalid entries
    probs_flatten: Float[Tensor, " k_groups_squared"] = probs.flatten()
    probs_flatten = probs_flatten / probs_flatten.sum()

    # Sample from multinomial distribution
    idx: int = int(torch.multinomial(probs_flatten, 1).item())
    row: int = idx // k_groups
    col: int = idx % k_groups

    return MergePair((row, col))


MERGE_PAIR_SAMPLERS: dict[MergePairSamplerKey, MergePairSamplerConfigurable] = {
    "range": range_sampler,
    "mcmc": mcmc_sampler,
}
