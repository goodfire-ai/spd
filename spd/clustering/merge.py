from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
import tqdm
import wandb
import wandb.sdk.wandb_run
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_pop_group,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.merge_run_config import MergeRunConfig
from spd.utils.wandb_tensor_info import wandb_log_tensor


def merge_iteration(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeRunConfig,
    component_labels: list[str],
    initial_merge: GroupMerge | None = None,
    sweep_params: dict[str, Any] | None = None,
    plot_function: Callable[..., None] | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    prefix: str = "",
    artifact_callback: Callable[[MergeHistory, int], None] | None = None,
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
        wandb_url=wandb_run.url if wandb_run else None,
    )

    # free up memory
    if not do_pop:
        del coact
        del activation_mask_orig
        del activations
        activation_mask_orig = None

    # merge iteration
    # ==================================================
    # while i < merge_config.iters:
    pbar = tqdm.tqdm(range(merge_config.iters), unit="iteration", total=merge_config.iters)
    for i in pbar:
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
        )

        # figure out what to merge, store some things
        # --------------------------------------------------

        # Sample a merge pair using the configured sampler
        merge_pair: tuple[int, int] = merge_config.merge_pair_sample(costs)

        # Update progress bar description with groups remaining and MDL loss
        mdl_loss = costs[merge_pair].mean().item()
        pbar.set_description(f"{prefix} k={k_groups}, mdl={mdl_loss:.4f}")

        # Store matrices and selected pair in history
        merge_history.add_iteration(
            idx=i,
            selected_pair=merge_pair,
            coactivation=current_coact,
            cost_matrix=costs,
            k_groups=k_groups,
            current_merge=current_merge,
        )

        # Log to WandB if enabled
        if wandb_run is not None and i % merge_config.wandb_log_frequency == 0:
            # Prepare additional stats
            group_sizes: Int[Tensor, " k_groups"] = current_merge.components_per_group
            fraction_singleton_groups: float = (group_sizes == 1).float().mean().item()
            group_sizes_no_singletons: Tensor = group_sizes[group_sizes > 1]

            fraction_zero_coacts: float = (current_coact == 0).float().mean().item()
            coact_no_zeros: Tensor = current_coact[current_coact != 0]
            diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)

            tensor_data_for_wandb: dict[str, Tensor] = dict(
                coactivation=current_coact,
                costs=costs,
                group_sizes=group_sizes,
                group_activations=diag_acts,
                group_activations_over_sizes=(
                    diag_acts / group_sizes.to(device=diag_acts.device).float()
                ),
            )

            if fraction_singleton_groups > 0:
                tensor_data_for_wandb["group_sizes_no_singletons"] = group_sizes_no_singletons
            if fraction_zero_coacts > 0:
                tensor_data_for_wandb["coact_no_zeros"] = coact_no_zeros

            wandb_log_tensor(
                run=wandb_run,
                data=tensor_data_for_wandb,
                name="iters",
                step=i,
            )

            # log chosen pair cost
            wandb_run.log(
                {
                    "iteration": i,
                    "k_groups": k_groups,
                    "merge_pair_cost": costs[merge_pair].item(),
                    "mdl_cost": compute_mdl_cost(
                        acts=diag_acts,
                        merges=current_merge,
                        alpha=merge_config.alpha,
                    ),
                    "fraction_singleton_groups": fraction_singleton_groups,
                    "fraction_zero_coacts": fraction_zero_coacts,
                }
            )

        # Call artifact callback periodically for saving group_idxs
        if (
            artifact_callback is not None
            and i > 0
            and i % merge_config.wandb_artifact_frequency == 0
        ):
            artifact_callback(merge_history, i)

        # merge the pair
        # --------------------------------------------------
        current_merge, current_coact, current_act_mask = recompute_coacts_merge_pair(
            coact=current_coact,
            merges=current_merge,
            merge_pair=merge_pair,
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
        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {i} as only {k_groups} groups left", stacklevel=1
            )
            break

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
