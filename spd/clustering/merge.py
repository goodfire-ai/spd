from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from typing import Any

import torch
import tqdm
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.compute_costs import (
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_pop_group,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble


def merge_iteration(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig,
    component_labels: list[str],
    initial_merge: GroupMerge | None = None,
    sweep_params: dict[str, Any] | None = None,
    plot_function: Callable | None = None,
) -> MergeHistory:
    # setup
    # ==================================================

    # compute coactivations
    activation_mask_orig: Float[Tensor, "samples c_components"] | None = (
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
    coact: Float[Tensor, "c_components c_components"] = (
        activation_mask_orig.float().T @ activation_mask_orig.float()
    )

    # check shapes
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"
    assert activation_mask_orig.shape[1] == c_components, (
        "Activation mask must match coactivation matrix shape"
    )

    # for speed, we precompute whether to pop components and which components to pop
    do_pop: bool = merge_config.pop_component_prob > 0.0
    if do_pop:
        iter_pop: Bool[Tensor, " iters"] = (
            torch.rand(merge_config.iters, device=coact.device) < merge_config.pop_component_prob
        )
        pop_component_idx: Int[Tensor, " iters"] = torch.randint(
            0, c_components, (merge_config.iters,), device=coact.device
        )

    # start with an identity merge
    current_merge: GroupMerge
    if initial_merge is not None:
        current_merge = initial_merge
    else:
        current_merge = GroupMerge.identity(n_components=c_components)

    # initialize variables for the merge process
    k_groups: int = c_components
    current_coact: Float[Tensor, "k_groups k_groups"] = coact.clone()
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask_orig.clone()
    i: int = 0

    # variables we keep track of
    merge_history: MergeHistory = MergeHistory.from_config(
        config=merge_config,
        c_components=c_components,
        component_labels=component_labels,
        sweep_params=sweep_params,
    )

    # free up memory
    if not do_pop:
        del coact
        del activation_mask_orig
        del activations
        activation_mask_orig = None

    # merge iteration
    # ==================================================
    while i < merge_config.iters:
        # pop components
        # --------------------------------------------------
        if do_pop and iter_pop[i]:  # pyright: ignore[reportPossiblyUnboundVariable]
            # we split up the group which our chosen component belongs to
            pop_component_idx_i: int = int(pop_component_idx[i].item())  # pyright: ignore[reportPossiblyUnboundVariable]
            components_in_pop_grp: int = int(
                current_merge.components_per_group[  # pyright: ignore[reportArgumentType]
                    current_merge.group_idxs[pop_component_idx_i].item()
                ]
            )

            # but, if the component is the only one in its group, there is nothing to do
            if components_in_pop_grp > 1:
                current_merge, current_coact, current_act_mask = recompute_coacts_pop_group(
                    coact=current_coact,
                    merges=current_merge,
                    component_idx=pop_component_idx_i,
                    activation_mask=current_act_mask,
                    # this complains if `activation_mask_orig is None`, but this is only the case
                    # if `do_pop` is False, which it won't be here. we do this to save memory
                    activation_mask_orig=activation_mask_orig,  # pyright: ignore[reportArgumentType]
                )
                k_groups = current_coact.shape[0]

        # compute costs
        # --------------------------------------------------
        # HACK: this is messy
        costs: Float[Tensor, "c_components c_components"] = compute_merge_costs(
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=merge_config.alpha,
            rank_cost=merge_config.rank_cost_fn,
        )

        # figure out what to merge, store some things
        # --------------------------------------------------

        # find the maximum cost among non-diagonal elements we should consider
        non_diag_costs: Float[Tensor, ""] = costs[~torch.eye(k_groups, dtype=torch.bool)]
        non_diag_costs_range: tuple[float, float] = (
            non_diag_costs.min().item(),
            non_diag_costs.max().item(),
        )
        max_considered_cost: float = (
            non_diag_costs_range[1] - non_diag_costs_range[0]
        ) * merge_config.check_threshold + non_diag_costs_range[0]

        # consider pairs with costs below the threshold
        considered_idxs = torch.where(costs <= max_considered_cost)
        considered_idxs = torch.stack(considered_idxs, dim=1)
        # remove from considered_idxs where i == j
        considered_idxs = considered_idxs[considered_idxs[:, 0] != considered_idxs[:, 1]]

        # randomly select one of the considered pairs
        min_pair: tuple[int, int] = tuple(
            considered_idxs[random.randint(0, considered_idxs.shape[0] - 1)].tolist()
        )
        pair_cost: float = costs[min_pair[0], min_pair[1]].item()

        # store for plotting
        merge_history.add_iteration(
            idx=i,
            non_diag_costs_range=non_diag_costs_range,
            max_considered_cost=max_considered_cost,
            pair_cost=pair_cost,
            k_groups=k_groups,
            current_merge=current_merge,
        )

        # merge the pair
        # --------------------------------------------------
        current_merge, current_coact, current_act_mask = recompute_coacts_merge_pair(
            coact=current_coact,
            merges=current_merge,
            merge_pair=min_pair,
            activation_mask=current_act_mask,
        )

        # iterate and sanity checks
        # --------------------------------------------------
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

        # plot if requested
        if plot_function is not None:
            plot_function(
                costs=costs,
                costs_computed=dict(
                    non_diag_costs=non_diag_costs,
                    non_diag_costs_range=non_diag_costs_range,
                    max_considered_cost=max_considered_cost,
                    considered_idxs=considered_idxs,
                    min_pair=min_pair,
                    pair_cost=pair_cost,
                ),
                merge_history=merge_history,
                current_merge=current_merge,
                current_coact=current_coact,
                current_act_mask=current_act_mask,
                i=i,
                k_groups=k_groups,
                activation_mask_orig=activation_mask_orig,
                component_labels=component_labels,
                sweep_params=sweep_params,
            )

        # early stopping
        # --------------------------------------------------

        # Check stopping conditions
        if k_groups <= 2:
            warnings.warn(
                f"Stopping early at iteration {i} as only {k_groups} groups left", stacklevel=1
            )
            break

        i += 1

    # finish up
    # ==================================================

    return merge_history


def merge_iteration_ensemble(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig,
    ensemble_size: int,
    component_labels: list[str],
    initial_merge: GroupMerge | None = None,
) -> MergeHistoryEnsemble:
    """Run many merge iterations"""

    output: list[MergeHistory] = []
    for _ in tqdm.tqdm(range(ensemble_size), unit="ensemble"):
        # run the merge iteration
        merge_history = merge_iteration(
            activations=activations,
            merge_config=merge_config,
            component_labels=component_labels,
            initial_merge=initial_merge,
        )

        # store the history
        output.append(merge_history)

    return MergeHistoryEnsemble(data=output)
