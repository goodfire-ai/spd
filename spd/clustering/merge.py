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

from spd.clustering.batched_activations import ActivationBatch, BatchedActivations
from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
)
from spd.clustering.consts import (
    ActivationsTensor,
    BoolActivationsTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
    MergePair,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory


def recompute_coacts_from_scratch(
    activations: Tensor,
    current_merge: GroupMerge,
    activation_threshold: float | None,
) -> tuple[Tensor, Tensor]:
    """
    Recompute coactivations from fresh activations using current merge state.

    Args:
        activations: Fresh activation tensor [samples, n_components_original]
        current_merge: Current merge state mapping original -> groups
        activation_threshold: Threshold for binarizing activations

    Returns:
        (coact, activation_mask) - coact matrix [k_groups, k_groups] and
                                   mask [samples, k_groups] for current groups
    """
    # Apply threshold
    activation_mask: Bool[Tensor, "samples n_components"] = (
        activations > activation_threshold if activation_threshold is not None else activations
    )

    # Map component-level activations to group-level using scatter_add
    # This is more efficient than materializing the full merge matrix
    # current_merge.group_idxs: [n_components] with values 0 to k_groups-1
    n_samples: int = activation_mask.shape[0]
    group_activations: Float[Tensor, "n_samples k_groups"] = torch.zeros(
        (n_samples, current_merge.k_groups),
        dtype=activation_mask.dtype,
        device=activation_mask.device,
    )

    # Expand group_idxs to match batch dimension and scatter-add activations by group
    group_idxs_expanded: Int[Tensor, "n_samples n_components"] = (
        current_merge.group_idxs.unsqueeze(0).expand(n_samples, -1).to(activation_mask.device)
    )
    group_activations.scatter_add_(1, group_idxs_expanded, activation_mask)

    # Compute coactivations
    coact: ClusterCoactivationShaped = group_activations.float().T @ group_activations.float()

    return coact, group_activations


class LogCallback(Protocol):
    def __call__(
        self,
        current_coact: ClusterCoactivationShaped,
        component_labels: ComponentLabels,
        current_merge: GroupMerge,
        costs: ClusterCoactivationShaped,
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
    batched_activations: BatchedActivations,
    component_labels: ComponentLabels,
    log_callback: LogCallback | None = None,
) -> MergeHistory:
    """
    Merge iteration with multi-batch support and optional logging/plotting callbacks.

    This implementation uses NaN masking to track invalid coactivation entries
    and periodically recomputes the full coactivation matrix from fresh batches.
    """

    # Load first batch
    # --------------------------------------------------
    first_batch: ActivationBatch = batched_activations._get_next_batch()
    activations: ActivationsTensor = first_batch.activations

    # Compute initial coactivations
    # --------------------------------------------------
    activation_mask_orig: BoolActivationsTensor | ActivationsTensor = (
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
    coact: Float[Tensor, "c c"] = activation_mask_orig.float().T @ activation_mask_orig.float()

    # check shapes
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"

    # determine number of iterations based on config and number of components
    num_iters: int = merge_config.get_num_iters(c_components)

    # initialize vars
    # --------------------------------------------------
    # start with an identity merge
    current_merge: GroupMerge = GroupMerge.identity(n_components=c_components)

    # initialize variables for the merge process
    k_groups: int = c_components
    current_coact: ClusterCoactivationShaped = coact.clone()
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask_orig.clone()

    # variables we keep track of
    merge_history: MergeHistory = MergeHistory.from_config(
        merge_config=merge_config,
        labels=component_labels,
    )

    # merge iteration
    # ==================================================
    pbar: tqdm[int] = tqdm(
        range(num_iters),
        unit="iter",
        total=num_iters,
    )
    for iter_idx in pbar:
        # Recompute from new batch if it's time (do this BEFORE computing costs)
        # --------------------------------------------------
        if merge_config.recompute_costs_every is not None:
            should_recompute: bool = (
                iter_idx % merge_config.recompute_costs_every == 0
            ) and (iter_idx > 0)

            if should_recompute:
                new_batch: ActivationBatch = batched_activations._get_next_batch()
                activations = new_batch.activations

                # Recompute fresh coacts with current merge groups
                current_coact, current_act_mask = recompute_coacts_from_scratch(
                    activations=activations,
                    current_merge=current_merge,
                    activation_threshold=merge_config.activation_threshold,
                )

        # compute costs, figure out what to merge
        # --------------------------------------------------
        # HACK: this is messy
        costs: ClusterCoactivationShaped = compute_merge_costs(
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=merge_config.alpha,
        )

        merge_pair: MergePair = merge_config.merge_pair_sample(costs)

        # Store merge pair cost before updating
        # --------------------------------------------------
        merge_pair_cost: float = float(costs[merge_pair].item())

        # merge the pair
        # --------------------------------------------------
        # Update merge state BEFORE NaN-ing out
        current_merge = current_merge.merge_groups(merge_pair[0], merge_pair[1])

        # NaN out the merged components' rows/cols
        i, j = merge_pair
        new_idx: int = min(i, j)
        remove_idx: int = max(i, j)

        # Mark affected entries as invalid (can't compute cost anymore without recompute)
        current_coact[remove_idx, :] = float("nan")
        current_coact[:, remove_idx] = float("nan")
        current_coact[new_idx, :] = float("nan")
        current_coact[:, new_idx] = float("nan")

        # Remove the deleted row/col to maintain shape consistency
        mask: Bool[Tensor, " k_groups"] = torch.ones(
            k_groups, dtype=torch.bool, device=current_coact.device
        )
        mask[remove_idx] = False
        current_coact = current_coact[mask, :][:, mask]
        current_act_mask = current_act_mask[:, mask]

        k_groups -= 1

        # Store in history
        # --------------------------------------------------
        merge_history.add_iteration(
            idx=iter_idx,
            selected_pair=merge_pair,
            current_merge=current_merge,
        )

        # Compute metrics for logging
        # --------------------------------------------------
        # the MDL loss computed here is the *cost of the current merge*, a single scalar value
        # rather than the *delta in cost from merging a specific pair* (which is what `costs` matrix contains)
        diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / current_act_mask.shape[0]

        # Update progress bar
        pbar.set_description(f"k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={merge_pair_cost:.4f}")

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

        # Sanity checks
        # --------------------------------------------------
        assert current_coact.shape[0] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_coact.shape[1] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_act_mask.shape[1] == k_groups, (
            "Activation mask shape should match number of groups"
        )

        # early stopping failsafe
        # --------------------------------------------------
        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {iter_idx} as only {k_groups} groups left",
                stacklevel=2,
            )
            break

    # finish up
    # ==================================================
    return merge_history
