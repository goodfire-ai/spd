from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor
import tqdm
from muutils.dbg import dbg_auto
from muutils.parallel import run_maybe_parallel

from spd.clustering.merge_matrix import GroupMerge
from spd.clustering.perm_invariant_hamming import perm_invariant_hamming_matrix
from spd.spd_types import Probability


def compute_merge_costs(
    coact: Bool[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
    rank_cost: Callable[[float], float] = lambda _: 1.0,
) -> Float[Tensor, "k_groups k_groups"]:
    """Compute MDL costs for merge matrices"""
    device: torch.device = coact.device
    ranks: Float[Tensor, " k_groups"] = merges.components_per_group.to(device=device).float()
    diag: Float[Tensor, " k_groups"] = torch.diag(coact).to(device=device)

    # TODO: use dynamic rank computation
    return alpha * (
        diag @ ranks.T
        + ranks @ diag.T
        - (ranks.unsqueeze(0) + ranks.unsqueeze(1) + (rank_cost(merges.k_groups) / alpha)) * coact
    )


def recompute_coacts_merge_pair(
    coact: Float[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    merge_pair: tuple[int, int],
    activation_mask: Bool[Tensor, "samples k_groups"],
) -> tuple[
    GroupMerge,
    Float[Tensor, "k_groups-1 k_groups-1"],
    Bool[Tensor, "samples k_groups"],
]:
    # check shape
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"

    # activations of the new merged group
    activation_mask_grp: Bool[Tensor, " samples"] = (
        activation_mask[:, merge_pair[0]] + activation_mask[:, merge_pair[1]]
    )

    # coactivations with the new merged group
    # dbg_tensor(activation_mask_grp)
    # dbg_tensor(activation_mask)
    coact_with_merge: Float[Tensor, " k_groups"] = (
        activation_mask_grp.float() @ activation_mask.float()
    )
    new_group_idx: int = min(merge_pair)
    remove_idx: int = max(merge_pair)
    new_group_self_coact: float = activation_mask_grp.float().sum().item()
    # dbg_tensor(coact_with_merge)

    # assemble the merge pair
    merge_new: GroupMerge = merges.merge_groups(
        merge_pair[0],
        merge_pair[1],
    )
    old_to_new_idx: dict[int | None, int | None] = merge_new.old_to_new_idx  # type: ignore
    assert old_to_new_idx[None] == new_group_idx, (
        "New group index should be the minimum of the merge pair"
    )
    assert old_to_new_idx[new_group_idx] is None
    assert old_to_new_idx[remove_idx] is None
    # TODO: check that the rest are in order? probably not necessary

    # reindex coactivations
    coact_temp: Float[Tensor, "k_groups k_groups"] = coact.clone()
    # add in the similarities with the new group
    coact_temp[new_group_idx, :] = coact_with_merge
    coact_temp[:, new_group_idx] = coact_with_merge
    # delete the old group
    mask: Bool[Tensor, " k_groups"] = torch.ones(
        coact_temp.shape[0], dtype=torch.bool, device=coact_temp.device
    )
    mask[remove_idx] = False
    coact_new: Float[Tensor, "k_groups-1 k_groups-1"] = coact_temp[mask, :][:, mask]
    # add in the self-coactivation of the new group
    coact_new[new_group_idx, new_group_idx] = new_group_self_coact
    # dbg_tensor(coact_new)

    # reindex mask
    activation_mask_new: Float[Tensor, "samples ..."] = activation_mask.clone()
    # add in the new group
    activation_mask_new[:, new_group_idx] = activation_mask_grp
    # remove the old group
    activation_mask_new = activation_mask_new[:, mask]

    # dbg_tensor(activation_mask_new)

    return (
        merge_new,
        coact_new,
        activation_mask_new,
    )


def recompute_coacts_pop_group(
    coact: Float[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    component_idx: int,
    activation_mask: Bool[Tensor, "n_samples k_groups"],
    activation_mask_orig: Bool[Tensor, "n_samples n_components"],
) -> tuple[
    GroupMerge,
    Float[Tensor, "k_groups+1 k_groups+1"],
    Bool[Tensor, "n_samples k_groups+1"],
]:
    # sanity check dims
    # ==================================================
    # dbg_tensor(coact)
    # dbg_tensor(activation_mask)
    # dbg_tensor(activation_mask_orig)

    k_groups: int = coact.shape[0]
    n_samples: int = activation_mask.shape[0]
    # n_components: int = activation_mask_orig.shape[1]
    k_groups_new: int = k_groups + 1
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert activation_mask.shape[1] == k_groups, (
        "Activation mask must match coactivation matrix shape"
    )
    assert n_samples == activation_mask_orig.shape[0], (
        "Activation mask original must match number of samples"
    )

    # get the activations we need
    # ==================================================
    # which group does the component belong to?
    group_idx: int = merges.group_idxs[component_idx].item()
    group_size_old: int = merges.components_per_group[group_idx].item()
    group_size_new: int = group_size_old - 1

    # activations of component we are popping out
    acts_pop: Bool[Tensor, " samples"] = activation_mask_orig[:, component_idx]

    # activations of the "remainder" -- everything other than the component we are popping out,
    # in the group we're popping it out of
    # dims: dict[str, int] = dict(
    # 	n_samples=n_samples,
    # 	k_groups=k_groups,
    # 	k_groups_new=k_groups_new,
    # 	group_size_old=group_size_old,
    # 	group_size_new=group_size_new,
    # 	n_components=n_components,
    # )
    # dbg_auto(dims)
    # dbg_tensor(acts_pop)
    # dbg_tensor(merges.components_in_group(group_idx))

    acts_remainder: Bool[Tensor, " samples"] = (
        activation_mask_orig[
            :, [i for i in merges.components_in_group(group_idx) if i != component_idx]
        ]
        .max(dim=-1)
        .values
    )

    # assemble the new activation mask
    # ==================================================
    # first concat the popped-out component onto the end
    activation_mask_new: Bool[Tensor, " samples k_groups+1"] = torch.cat(
        [activation_mask, acts_pop.unsqueeze(1)],
        dim=1,
    )
    # then replace the group we are popping out of with the remainder
    activation_mask_new[:, group_idx] = acts_remainder

    # dbg_tensor(acts_remainder)
    # dbg_tensor(activation_mask_new)

    # assemble the new coactivation matrix
    # ==================================================
    coact_new: Float[Tensor, "k_groups+1 k_groups+1"] = torch.full(
        (k_groups_new, k_groups_new),
        fill_value=float("nan"),
        dtype=coact.dtype,
        device=coact.device,
    )
    # copy in the old coactivation matrix
    coact_new[:k_groups, :k_groups] = coact.clone()
    # compute new coactivations we need
    coact_pop: Float[Tensor, " k_groups"] = acts_pop.float() @ activation_mask_new.float()
    coact_remainder: Float[Tensor, " k_groups"] = (
        acts_remainder.float() @ activation_mask_new.float()
    )

    # dbg_tensor(coact)
    # dbg_tensor(coact_new)
    # dbg_tensor(coact_pop)
    # dbg_tensor(coact_remainder)

    # replace the relevant rows and columns
    coact_new[group_idx, :] = coact_remainder
    coact_new[:, group_idx] = coact_remainder
    coact_new[-1, :] = coact_pop
    coact_new[:, -1] = coact_pop

    # assemble the new group merge
    # ==================================================
    group_idxs_new: Int[Tensor, " k_groups+1"] = merges.group_idxs.clone()
    # the popped-out component is now its own group
    new_group_idx: int = k_groups_new - 1
    group_idxs_new[component_idx] = new_group_idx
    merge_new: GroupMerge = GroupMerge(
        group_idxs=group_idxs_new,
        k_groups=k_groups_new,
    )

    # sanity check
    assert merge_new.components_per_group.shape == (k_groups_new,), (
        "New merge must have k_groups+1 components"
    )
    assert merge_new.components_per_group[new_group_idx] == 1, (
        "New group must have exactly one component"
    )
    assert merge_new.components_per_group[group_idx] == group_size_new, (
        "Old group must have one less component"
    )

    # return
    # ==================================================
    return (
        merge_new,
        coact_new,
        activation_mask_new,
    )


class MergeConfig(BaseModel):
    activation_threshold: Probability | None = Field(
        default=0.01,
        description="Threshold for considering a component active in a group. If None, use raw scalar causal importances",
    )
    alpha: float = Field(
        default=1.0,
        description="rank weight factor. Higher values mean a higher penalty on 'sending' the component weights",
    )
    iters: PositiveInt = Field(
        default=100,
        description="max number of iterations to run the merge algorithm for.",
    )
    check_threshold: Probability = Field(
        default=0.05,
        description="threshold for considering merge pairs, as a fraction of the range of non-diagonal costs. If 0, always select the pair with the lowest cost. if 1, choose randomly among all pairs.",
    )
    pop_component_prob: Probability = Field(
        default=0.0,
        description="Probability of popping a component in each iteration. If 0, no components are popped.",
    )

    rank_cost_fn: Callable[[float], float] = lambda _: 1.0
    stopping_condition: Callable[[MergeHistory], bool] | None = None

    @property
    def rank_cost_name(self) -> str:
        """Get the name of the rank cost function."""
        return getattr(self.rank_cost_fn, "__name__", str(self.rank_cost_fn))


class MergePlotConfig(BaseModel):
    plot_every: int = 20
    plot_every_min: int = 0
    save_pdf: bool = False
    pdf_prefix: str = "merge_iteration"
    figsize: tuple[int, int] = (16, 3)
    figsize_final: tuple[int, int] = (10, 6)
    tick_spacing: int = 10
    plot_final: bool = True


class MergeHistory:
    """Track merge iteration history with validation."""

    c_components: int
    non_diag_costs_min: list[float]
    non_diag_costs_max: list[float]
    max_considered_cost: list[float]
    selected_pair_cost: list[float]
    costs_range: list[float]
    k_groups: list[int]
    merges: list[GroupMerge]  # State of groups at each iteration
    config: MergeConfig | None = None  # Configuration used for this merge
    sweep_params: dict[str, Any] | None = None  # Sweep parameters if used in sweep

    def __init__(self, config: MergeConfig, c_components: int) -> None:
        self.c_components = c_components
        self.non_diag_costs_min = []
        self.non_diag_costs_max = []
        self.max_considered_cost = []
        self.selected_pair_cost = []
        self.costs_range = []
        self.k_groups = []
        self.merges = []
        self.config = config
        self.sweep_params = None

    def __post_init__(self) -> None:
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        """Ensure all lists have the same length."""
        lengths = [
            len(self.non_diag_costs_min),
            len(self.non_diag_costs_max),
            len(self.max_considered_cost),
            len(self.selected_pair_cost),
            len(self.costs_range),
            len(self.k_groups),
            len(self.merges),
        ]
        if lengths and not all(length == lengths[0] for length in lengths):
            raise ValueError("All history lists must have the same length")

    def add_iteration(
        self,
        non_diag_costs_range: tuple[float, float],
        max_considered_cost: float,
        pair_cost: float,
        k_groups: int,
        current_merge: GroupMerge,
    ) -> None:
        """Add data for one iteration."""
        self.non_diag_costs_min.append(non_diag_costs_range[0])
        self.non_diag_costs_max.append(non_diag_costs_range[1])
        self.max_considered_cost.append(max_considered_cost)
        self.costs_range.append(non_diag_costs_range[1] - non_diag_costs_range[0])
        self.selected_pair_cost.append(pair_cost)
        self.k_groups.append(k_groups)
        self.merges.append(current_merge)
        self._validate_lengths()

    def latest(self) -> dict[str, float | int]:
        """Get the latest values."""
        if not self.non_diag_costs_min:
            raise ValueError("No history available")
        return {
            "non_diag_costs_min": self.non_diag_costs_min[-1],
            "non_diag_costs_max": self.non_diag_costs_max[-1],
            "max_considered_cost": self.max_considered_cost[-1],
            "selected_pair_cost": self.selected_pair_cost[-1],
            "costs_range": self.costs_range[-1],
            "k_groups": self.k_groups[-1],
        }

    def plot(self, plot_config: MergePlotConfig | None = None) -> None:
        """Plot cost evolution."""
        from spd.clustering.plotting.merge import plot_merge_history
        plot_merge_history(self, plot_config)

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        """Convert to dictionary for backward compatibility."""
        return {
            "non_diag_costs_min": self.non_diag_costs_min,
            "non_diag_costs_max": self.non_diag_costs_max,
            "max_considered_cost": self.max_considered_cost,
            "selected_pair_cost": self.selected_pair_cost,
            "costs_range": self.costs_range,
            "k_groups": self.k_groups,
        }

    # Convenience properties for sweep analysis
    @property
    def total_iterations(self) -> int:
        """Total number of iterations performed."""
        return len(self.non_diag_costs_min)

    @property
    def final_k_groups(self) -> int:
        """Final number of groups after merging."""
        return self.k_groups[-1] if self.k_groups else 0

    @property
    def initial_k_groups(self) -> int:
        """Initial number of groups before merging."""
        return self.k_groups[0] if self.k_groups else 0




def merge_iteration(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig,
    component_labels: list[str] | None = None,
    initial_merge: GroupMerge | None = None,
    plot_config: MergePlotConfig | None = None,
) -> tuple[MergeHistory, GroupMerge]:
    # setup
    # ==================================================

    # compute coactivations
    activation_mask: Float[Tensor, "samples c_components"] = (
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
    coact: Float[Tensor, "c_components c_components"] = (
        activation_mask.float().T @ activation_mask.float()
    )

    # check shapes
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"
    assert activation_mask.shape[1] == c_components, (
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
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask.clone()
    i: int = 0

    # variables we keep track of
    merge_history = MergeHistory(config=merge_config, c_components=c_components)

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
                    activation_mask_orig=activation_mask,
                )
                k_groups = current_coact.shape[0]

        # compute costs
        # --------------------------------------------------
        costs: Float[Tensor, "c_components c_components"] = compute_merge_costs(
            coact=current_coact,
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

        # early stopping
        # --------------------------------------------------

        # Check stopping conditions
        if k_groups <= 2:
            warnings.warn(
                f"Stopping early at iteration {i} as only {k_groups} groups left", stacklevel=1
            )
            current_merge.plot(component_labels=component_labels)
            break

        # Custom stopping condition
        if merge_config.stopping_condition is not None:  # noqa: SIM102
            if merge_config.stopping_condition(merge_history):
                break

        i += 1

    # finish up
    # ==================================================

    return merge_history, current_merge




@dataclass
class MergeEnsemble:
    data: list[MergeHistory]

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, idx: int) -> MergeHistory:
        return self.data[idx]
    
    @property
    def n_iters(self) -> int:
        """Number of iterations in the ensemble."""
        n_iterations: int = len(self.data[0].k_groups)
        assert all(
            len(history.k_groups) == n_iterations for history in self.data
        ), "All histories must have the same number of iterations"
        return n_iterations
    
    @property
    def n_ensemble(self) -> int:
        """Number of ensemble members."""
        return len(self.data)

    @property
    def merges_array(self) -> Int[np.ndarray, "n_ens n_iters c_components"]:
        n_ens: int = self.n_ensemble
        n_iters: int = self.n_iters
        c_components: int = self.data[0].c_components
        assert all(
            h.c_components == c_components for h in self.data
        ), "All histories must have the same number of components"

        output: Int[np.ndarray, "n_ens n_iters c_components"] = np.full(
            (n_ens, n_iters, c_components),
            fill_value=-1,
            dtype=np.int16,
            # if you have more than 32k components, change this to np.int32
            # if you have more than 2.1b components, rethink your life choices
        )
        for i_ens, history in enumerate(self.data):
            for i_iter, merge in enumerate(history.merges):
                output[i_ens, i_iter] = merge.group_idxs

        return output


    def get_distances(self) -> Float[np.ndarray, "n_iters n_ens n_ens"]:
        n_iters: int = self.n_iters
        n_ens: int = self.n_ensemble

        merges_array: Int[np.ndarray, "n_ens n_iters c_components"] = self.merges_array
        # for i in tqdm.tqdm(range(n_iters)):
        #     distances[i] = perm_invariant_hamming_matrix(
        #         merges_array[:, i, :]
        #     )
        merges_array_list: list[Int[np.ndarray, "n_ens c_components"]] = [
            merges_array[:, i, :] for i in range(n_iters)
        ]

        distances_list: list[Float[np.ndarray, "n_ens n_ens"]] = run_maybe_parallel(
            func=perm_invariant_hamming_matrix,
            iterable=merges_array_list,
            parallel=True,
        )

        return np.stack(distances_list, axis=0)
    



def merge_iteration_ensemble(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig,
    ensemble_size: int,
    component_labels: list[str] | None = None,
    initial_merge: GroupMerge | None = None,
) -> MergeEnsemble:
    """Run many merge iterations"""

    output: list[MergeHistory] = []
    for _ in tqdm.tqdm(range(ensemble_size), unit="ensemble"):
        # run the merge iteration
        merge_history, _ = merge_iteration(
            activations=activations,
            merge_config=merge_config,
            component_labels=component_labels,
            initial_merge=initial_merge,
        )

        # store the history
        output.append(merge_history)

    return MergeEnsemble(data=output)




