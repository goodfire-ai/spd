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
        costs: Cost matrix for all possible merges
        k_groups: Number of current groups
        threshold: Fraction of cost range to consider (0=min only, 1=all pairs)

    Returns:
        Tuple of (group_i, group_j) indices to merge
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    # Find the range of non-diagonal costs
    non_diag_costs: Float[Tensor, " k_groups_squared_minus_k"] = costs[
        ~torch.eye(k_groups, dtype=torch.bool, device=costs.device)
    ]
    min_cost: float = float(non_diag_costs.min().item())
    max_cost: float = float(non_diag_costs.max().item())

    # Calculate threshold cost
    max_considered_cost: float = (max_cost - min_cost) * threshold + min_cost

    # Find all pairs below threshold
    considered_idxs: Int[Tensor, "n_considered 2"] = torch.stack(
        torch.where(costs <= max_considered_cost), dim=1
    )
    # Remove diagonal entries (i == j)
    considered_idxs = considered_idxs[considered_idxs[:, 0] != considered_idxs[:, 1]]

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
        costs: Cost matrix for all possible merges
        k_groups: Number of current groups
        temperature: Temperature parameter for softmax (higher = more uniform sampling)

    Returns:
        Tuple of (group_i, group_j) indices to merge
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    # Create mask for valid pairs (non-diagonal)
    valid_mask: Bool[Tensor, "k_groups k_groups"] = ~torch.eye(
        k_groups, dtype=torch.bool, device=costs.device
    )

    # Compute probabilities: exp(-cost/temperature)
    # Use stable softmax computation to avoid overflow
    costs_masked: ClusterCoactivationShaped = costs.clone()
    costs_masked[~valid_mask] = float("inf")  # Set diagonal to inf so exp gives 0

    # Subtract min for numerical stability
    min_cost: float = float(costs_masked[valid_mask].min())
    probs: ClusterCoactivationShaped = (
        torch.exp((min_cost - costs_masked) / temperature) * valid_mask
    )  # Zero out diagonal
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
