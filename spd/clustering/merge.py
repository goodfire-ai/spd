"""
Merge iteration with logging support.

This wraps the pure merge_iteration_pure() function and adds WandB/plotting callbacks.
"""

import warnings
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_pop_group,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory


class LogCallback(Protocol):
    def __call__(
        self,
        current_coact: Float[Tensor, "k_groups k_groups"],
        component_labels: list[str],
        current_merge: GroupMerge,
        costs: Float[Tensor, "k_groups k_groups"],
        merge_history: MergeHistory,
        iter_idx: int,
        k_groups: int,
        merge_pair_cost: float,
        mdl_loss: float,
        mdl_loss_norm: float,
        diag_acts: Float[Tensor, " k_groups"],
    ) -> None: ...


def merge_iteration(
    merge_config: MergeConfig,
    batch_id: str,
    activations: Float[Tensor, "n_steps c"],
    component_labels: list[str],
    log_callback: LogCallback | None = None,
) -> MergeHistory:
    """
    Merge iteration with optional logging/plotting callbacks.

    This wraps the pure computation with logging capabilities while maintaining
    the same core algorithm logic.
    """

    # Compute coactivations
    activation_mask_orig: Bool[Tensor, "n_steps c"] | Float[Tensor, "n_steps c"] | None = (
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
    coact: Float[Tensor, "c c"] = activation_mask_orig.float().T @ activation_mask_orig.float()

    # Setup
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"

    # Prepare pop component logic
    do_pop = merge_config.pop_component_prob > 0.0
    if do_pop:
        iter_pop = (
            torch.rand(merge_config.iters, device=coact.device) < merge_config.pop_component_prob
        )
        pop_component_idx = torch.randint(
            0, c_components, (merge_config.iters,), device=coact.device
        )

    # for speed, we precompute whether to pop components and which components to pop
    # if we are not popping, we don't need these variables and can also delete other things
    do_pop: bool = merge_config.pop_component_prob > 0.0
    if do_pop:
        # at each iteration, we will pop a component with probability `pop_component_prob`
        iter_pop: Bool[Tensor, " iters"] = (
            torch.rand(merge_config.iters, device=coact.device) < merge_config.pop_component_prob
        )
        # we pick a subcomponent at random, and if we decide to pop, we pop that one out of its group
        # if the component is a singleton, nothing happens. this naturally biases towards popping
        # less at the start and more at the end, since the effective probability of popping a component
        # is actually something like `pop_component_prob * (c_components - k_groups) / c_components`
        pop_component_idx: Int[Tensor, " iters"] = torch.randint(
            0, c_components, (merge_config.iters,), device=coact.device
        )

    # Initialize merge
    current_merge: GroupMerge = GroupMerge.identity(n_components=c_components)

    # Initialize variables
    k_groups: int = c_components
    current_coact: Float[Tensor, "k_groups k_groups"] = coact.clone()
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask_orig.clone()

    # Initialize history
    merge_history: MergeHistory = MergeHistory.from_config(
        config=merge_config, labels=component_labels
    )

    # Memory cleanup
    if not do_pop:
        del coact
        del activation_mask_orig
        activation_mask_orig = None

    # Main iteration loop with progress bar
    pbar: tqdm[int] = tqdm(range(merge_config.iters), unit="iter", total=merge_config.iters)
    for iter_idx in pbar:
        if do_pop and iter_pop[iter_idx]:  # pyright: ignore[reportPossiblyUnboundVariable]
            assert activation_mask_orig is not None, "Activation mask original is None"

            pop_component_idx_i: int = int(pop_component_idx[iter_idx].item())  # pyright: ignore[reportPossiblyUnboundVariable]
            group_idx: int = int(current_merge.group_idxs[pop_component_idx_i].item())
            n_components_in_pop_grp: int = int(current_merge.components_per_group[group_idx].item())

            if n_components_in_pop_grp > 1:
                current_merge, current_coact, current_act_mask = recompute_coacts_pop_group(
                    coact=current_coact,
                    merges=current_merge,
                    component_idx=pop_component_idx_i,
                    activation_mask=current_act_mask,
                    activation_mask_orig=activation_mask_orig,
                )
                k_groups = current_coact.shape[0]

        # Compute costs
        costs: Float[Tensor, "k_groups k_groups"] = compute_merge_costs(
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=merge_config.alpha,
        )

        merge_pair: tuple[int, int] = merge_config.merge_pair_sample(costs)

        # Merge the pair (after logging so we can see the cost)
        current_merge, current_coact, current_act_mask = recompute_coacts_merge_pair(
            coact=current_coact,
            merges=current_merge,
            merge_pair=merge_pair,
            activation_mask=current_act_mask,
        )

        # Store in history
        merge_history.add_iteration(
            idx=iter_idx,
            selected_pair=merge_pair,
            current_merge=current_merge,
        )

        # Compute metrics for logging
        diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / current_act_mask.shape[0]
        merge_pair_cost: float = float(costs[merge_pair].item())

        # Update progress bar

        prefix: str = f"\033[38;5;208m[{batch_id}]\033[0m"
        pbar.set_description(
            f"{prefix} k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={merge_pair_cost:.4f}"
        )

        if log_callback is not None:
            log_callback(
                iter_idx=iter_idx,
                current_coact=current_coact,
                component_labels=component_labels,
                current_merge=current_merge,
                costs=costs,
                merge_history=merge_history,
                k_groups=k_groups,
                merge_pair_cost=merge_pair_cost,
                mdl_loss=mdl_loss,
                mdl_loss_norm=mdl_loss_norm,
                diag_acts=diag_acts,
            )

        # Update and check
        k_groups -= 1
        assert current_coact.shape[0] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_coact.shape[1] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_act_mask.shape[1] == k_groups, (
            "Activation mask shape should match number of groups"
        )

        # Early stopping
        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {iter_idx} as only {k_groups} groups left",
                stacklevel=2,
            )
            break

    return merge_history
