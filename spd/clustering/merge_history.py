from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from muutils.dbg import dbg_tensor
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from torch import Tensor

from spd.clustering.math.merge_distances import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
    compute_distances,
)
from spd.clustering.math.merge_matrix import BatchedGroupMerge, GroupMerge
from spd.clustering.merge_config import MergeConfig


# pyright hates muutils :(
@serializable_dataclass(kw_only=True)  # pyright: ignore[reportUntypedClassDecorator]
class MergeHistory(SerializableDataclass):
    """Track merge iteration history"""

    c_components: int
    component_labels: list[str]
    n_iters_current: int
    non_diag_costs_min: Float[Tensor, " n_iters"]
    non_diag_costs_max: Float[Tensor, " n_iters"]
    max_considered_cost: Float[Tensor, " n_iters"]
    selected_pair_cost: Float[Tensor, " n_iters"]
    costs_range: Float[Tensor, " n_iters"]
    k_groups: Int[Tensor, " n_iters"]
    merges: BatchedGroupMerge = serializable_field(
        serialization_fn=lambda x: x.serialize(),
        deserialize_fn=lambda x: BatchedGroupMerge.load(x),
    )
    "State of groups at each iteration"

    config: MergeConfig = serializable_field(
        serialization_fn=lambda x: x.model_dump(mode="json"),
        deserialize_fn=lambda x: MergeConfig.model_validate(x),
    )
    "Configuration used for this merge"

    sweep_params: dict[str, Any] | None = serializable_field(
        default=None,
    )
    "Sweep parameters if used in sweep"

    @classmethod
    def from_config(
        cls,
        config: MergeConfig,
        c_components: int,
        component_labels: list[str],
        sweep_params: dict[str, Any] | None = None,
    ) -> MergeHistory:
        n_iters_target: int = config.iters
        # TODO: pyright doesnt like muutils
        return MergeHistory(
            c_components=c_components,  # pyright: ignore[reportCallIssue]
            component_labels=component_labels,  # pyright: ignore[reportCallIssue]
            n_iters_current=0,  # pyright: ignore[reportCallIssue]
            non_diag_costs_min=torch.full((n_iters_target,), float("nan"), dtype=torch.float32),  # pyright: ignore[reportCallIssue]
            non_diag_costs_max=torch.full((n_iters_target,), float("nan"), dtype=torch.float32),  # pyright: ignore[reportCallIssue]
            max_considered_cost=torch.full((n_iters_target,), float("nan"), dtype=torch.float32),  # pyright: ignore[reportCallIssue]
            selected_pair_cost=torch.full((n_iters_target,), float("nan"), dtype=torch.float32),  # pyright: ignore[reportCallIssue]
            costs_range=torch.full((n_iters_target,), float("nan"), dtype=torch.float32),  # pyright: ignore[reportCallIssue]
            k_groups=torch.full((n_iters_target,), -1, dtype=torch.int16),  # pyright: ignore[reportCallIssue]
            merges=BatchedGroupMerge.init_empty(  # pyright: ignore[reportCallIssue]
                batch_size=n_iters_target, n_components=c_components
            ),
            config=config,  # pyright: ignore[reportCallIssue]
            sweep_params=sweep_params,  # pyright: ignore[reportCallIssue]
        )

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
        idx: int,
        non_diag_costs_range: tuple[float, float],
        max_considered_cost: float,
        pair_cost: float,
        k_groups: int,
        current_merge: GroupMerge,
    ) -> None:
        """Add data for one iteration."""
        self.non_diag_costs_min[idx] = non_diag_costs_range[0]
        self.non_diag_costs_max[idx] = non_diag_costs_range[1]
        self.max_considered_cost[idx] = max_considered_cost
        self.costs_range[idx] = non_diag_costs_range[1] - non_diag_costs_range[0]
        self.selected_pair_cost[idx] = pair_cost
        self.k_groups[idx] = k_groups
        self.merges[idx] = current_merge

        assert self.n_iters_current == idx
        self.n_iters_current += 1

    def __getitem__(self, idx: int) -> dict[str, float | int | GroupMerge]:
        """Get data for a specific iteration."""
        if idx < 0 or idx >= self.n_iters_current:
            raise IndexError(f"Index {idx} out of range for history with {self.n_iters_current} iterations")
        return dict(
            non_diag_costs_min=self.non_diag_costs_min[idx].item(),
            non_diag_costs_max=self.non_diag_costs_max[idx].item(),
            max_considered_cost=self.max_considered_cost[idx].item(),
            selected_pair_cost=self.selected_pair_cost[idx].item(),
            costs_range=self.costs_range[idx].item(),
            k_groups=self.k_groups[idx].item(),
            merges=self.merges[idx],
        )
    
    def __len__(self) -> int:
        """Get the number of iterations in the history."""
        return self.n_iters_current
    
    def latest(self) -> dict[str, float | int | GroupMerge]:
        """Get the latest values."""
        if not len(self.non_diag_costs_min):
            raise ValueError("No history available")
        latest_idx: int = self.n_iters_current - 1
        return self[latest_idx]

    # Convenience properties for sweep analysis
    @property
    def total_iterations(self) -> int:
        """Total number of iterations performed."""
        return len(self.non_diag_costs_min)

    @property
    def final_k_groups(self) -> int:
        """Final number of groups after merging."""
        # return self.k_groups[-1] if self.k_groups else 0
        return int(self.k_groups[self.n_iters_current - 1].item())

    @property
    def initial_k_groups(self) -> int:
        """Initial number of groups before merging."""
        return int(self.k_groups[0].item())


@dataclass
class MergeHistoryEnsemble:
    data: list[MergeHistory]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> MergeHistory:
        return self.data[idx]

    def _validate_configs_match(self) -> None:
        """Ensure all histories have the same merge config."""
        if not self.data:
            return
        first_config: MergeConfig = self.data[0].config
        for history in self.data[1:]:
            if history.config != first_config:
                raise ValueError("All histories must have the same merge config")

    @property
    def config(self) -> MergeConfig:
        """Get the merge config used in the ensemble."""
        self._validate_configs_match()
        return self.data[0].config

    @property
    def n_iters(self) -> int:
        """Number of iterations in the ensemble."""
        n_iterations: int = len(self.data[0].k_groups)
        assert all(len(history.k_groups) == n_iterations for history in self.data), (
            "All histories must have the same number of iterations"
        )
        return n_iterations

    @property
    def n_ensemble(self) -> int:
        """Number of ensemble members."""
        return len(self.data)

    @property
    def c_components(self) -> int:
        """Number of components in each history."""
        c_components: int = self.data[0].c_components
        assert all(history.c_components == c_components for history in self.data), (
            "All histories must have the same number of components"
        )
        return c_components

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the ensemble data."""
        return (self.n_ensemble, self.n_iters, self.c_components)

    @property
    def merges_array(self) -> MergesArray:
        n_ens: int = self.n_ensemble
        n_iters: int = self.n_iters
        c_components: int = self.c_components

        output: MergesArray = np.full(
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

    def normalized(self) -> tuple[MergesArray, dict[str, Any]]:
        """Normalize the component labels across all histories.

        if different histories see different batches, then they might have different dead
        components, and are hence not directly comparable. So, we find the union of all
        component labels across all histories, and then any component missing from a history
        is put into it's own group in that history
        """

        unique_labels_set: set[str] = set()
        for history in self.data:
            unique_labels_set.update(history.component_labels)

        unique_labels: list[str] = sorted(unique_labels_set)
        c_components: int = len(unique_labels)
        component_label_idxs: dict[str, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }

        merges_array: MergesArray = np.full(
            (self.n_ensemble, self.n_iters, c_components),
            fill_value=-1,
            dtype=np.int16,
        )

        overlap_stats: Float[np.ndarray, " n_ens"] = np.full(
            self.n_ensemble,
            fill_value=float("nan"),
            dtype=np.float32,
        )
        i_ens: int
        history: MergeHistory
        for i_ens, history in enumerate(self.data):
            hist_c_labels: list[str] = history.component_labels
            hist_n_components: int = len(hist_c_labels)
            overlap_stats[i_ens] = hist_n_components / c_components
            # map from old component indices to new component indices
            for i_comp_old, comp_label in enumerate(hist_c_labels):
                i_comp_new: int = component_label_idxs[comp_label]
                merges_array[i_ens, :, i_comp_new] = history.merges.group_idxs[:, i_comp_old]

            assert np.max(merges_array[i_ens]) == hist_n_components - 1, (
                f"Max component index in history {i_ens} should be {hist_n_components - 1}, "
                f"but got {np.max(merges_array[i_ens])}"
            )

            # put each missing label into its own group
            hist_missing_labels: set[str] = unique_labels_set - set(hist_c_labels)
            assert len(hist_missing_labels) == c_components - hist_n_components
            for idx_missing, missing_label in enumerate(hist_missing_labels):
                i_comp_new_relabel: int = component_label_idxs[missing_label]
                merges_array[i_ens, :, i_comp_new_relabel] = np.full(
                    self.n_iters,
                    fill_value=idx_missing + hist_n_components,
                    dtype=np.int16,
                )

        dbg_tensor(overlap_stats)

        return (
            merges_array,
            dict(
                component_labels=unique_labels,
                n_ensemble=self.n_ensemble,
                n_iters=self.n_iters,
                c_components=c_components,
                config=self.config.model_dump(mode="json"),
            ),
        )

    def get_distances(self, method: DistancesMethod = "perm_invariant_hamming") -> DistancesArray:
        _n_iters: int = self.n_iters
        _n_ens: int = self.n_ensemble

        merges_array: MergesArray = self.merges_array
        return compute_distances(
            normalized_merge_array=merges_array,
            method=method,
        )
