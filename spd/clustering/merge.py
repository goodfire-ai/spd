<<<<<<< HEAD
"""
Merge iteration with logging support.

This wraps the pure merge_iteration_pure() function and adds WandB/plotting callbacks.
"""

import warnings
from typing import Protocol

import torch
=======
import warnings
from collections.abc import Callable

import torch
import wandb
import wandb.sdk.wandb_run
>>>>>>> chinyemba/feature/clustering-sjcs
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_pop_group,
)
<<<<<<< HEAD
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

_BATCH_PREFIX_FMT: str = "\033[38;5;208m[{batch_id}]\033[0m"


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
    activations: ActivationsTensor,
    component_labels: ComponentLabels,
    log_callback: LogCallback | None = None,
    batch_id: str = "unk",
) -> MergeHistory:
    """
    Merge iteration with optional logging/plotting callbacks.

    This wraps the pure computation with logging capabilities while maintaining
    the same core algorithm logic.
    """

    # setup
    # ==================================================
    pbar_prefix: str = _BATCH_PREFIX_FMT.format(batch_id=batch_id)

    # compute coactivations
    # --------------------------------------------------
    activation_mask_orig: BoolActivationsTensor | ActivationsTensor | None = (
=======
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.merge_run_config import _DEFAULT_INTERVALS, IntervalsDict, MergeRunConfig
from spd.clustering.wandb_tensor_info import wandb_log_tensor


def _wandb_iter_log(
    # general
    wandb_run: wandb.sdk.wandb_run.Run | None,
    merge_config: MergeConfig | MergeRunConfig,
    merge_history: MergeHistory,
    component_labels: list[str],
    # dims
    iter_idx: int,
    k_groups: int,
    n_samples: int,
    # actual data
    current_merge: GroupMerge,
    merge_pair: tuple[int, int],
    costs: Float[Tensor, "k_groups k_groups"],
    current_coact: Float[Tensor, "k_groups k_groups"],
    # progress bar stuff
    prefix: str,
    pbar: "tqdm[int]",
    # callbacks
    artifact_callback: Callable[[MergeHistory, int], None] | None = None,
    plot_function: Callable[..., None] | None = None,
    # config
    semilog_epsilon: float = 1e-3,
) -> None:
    """store in merge history, log to wandb, update progress bar, save artifacts, and make plots"""
    intervals: IntervalsDict = getattr(merge_config, "intervals", _DEFAULT_INTERVALS)
    # compute things we need to log
    # ============================================================
    diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
    # the MDL loss computed here is the *cost of the current merge*, a single scalar value
    # rather than the *delta in cost from merging a specific pair* (which is what `costs` matrix contains)
    mdl_loss: float = compute_mdl_cost(
        acts=diag_acts,
        merges=current_merge,
        alpha=merge_config.alpha,
    )
    mdl_loss_norm: float = mdl_loss / n_samples
    pbar.set_description(
        f"{prefix} k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={float(costs[merge_pair].item()):.4f}"
    )
    # this is the cost for the selected pair
    merge_pair_cost: float = float(costs[merge_pair].item())
    merge_pair_cost_semilog: float = semilog(
        value=merge_pair_cost,
        epsilon=semilog_epsilon,
    )

    # Store matrices and selected pair in history
    # ============================================================
    merge_history.add_iteration(
        idx=iter_idx,
        selected_pair=merge_pair,
        current_merge=current_merge,
    )

    # Log to WandB if enabled
    # ============================================================
    if wandb_run is not None:
        # Log basic stats at "stat" interval
        if iter_idx % intervals["stat"] == 0:
            wandb_run.log(
                {
                    "k_groups": int(k_groups),
                    "merge_pair_cost": merge_pair_cost,
                    f"merge_pair_cost_semilog[{semilog_epsilon}]": merge_pair_cost_semilog,
                    "mdl_loss": float(mdl_loss),
                    "mdl_loss_norm": float(mdl_loss_norm),
                },
                step=iter_idx,
            )

        # Log tensors and fraction stats at "tensor" interval
        if iter_idx % intervals["tensor"] == 0:
            # Prepare additional stats
            group_sizes: Int[Tensor, " k_groups"] = current_merge.components_per_group
            fraction_singleton_groups: float = (group_sizes == 1).float().mean().item()
            group_sizes_log1p: Tensor = torch.log1p(group_sizes.float())

            fraction_zero_coacts: float = (current_coact == 0).float().mean().item()
            coact_log1p: Tensor = torch.log1p(current_coact.float())

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
                tensor_data_for_wandb["group_sizes.log1p"] = group_sizes_log1p
            if fraction_zero_coacts > 0:
                tensor_data_for_wandb["coactivation.log1p"] = coact_log1p

            # log the tensors -- this makes histograms, and also stats about the tensors in tensor_metrics
            wandb_log_tensor(
                run=wandb_run,
                data=tensor_data_for_wandb,
                name="iters",
                step=iter_idx,
            )

            # Also log the fraction stats
            wandb_run.log(
                {
                    "fraction_singleton_groups": float(fraction_singleton_groups),
                    "fraction_zero_coacts": float(fraction_zero_coacts),
                },
                step=iter_idx,
            )

    # Call artifact callback periodically for saving group_idxs
    # ============================================================
    if artifact_callback is not None and iter_idx > 0 and iter_idx % intervals["artifact"] == 0:
        artifact_callback(merge_history, iter_idx)

    # plot if requested
    # ============================================================
    if plot_function is not None and iter_idx % intervals["plot"] == 0:
        plot_function(
            costs=costs,
            merge_history=merge_history,
            current_merge=current_merge,
            current_coact=current_coact,
            i=iter_idx,
            k_groups=k_groups,
            component_labels=component_labels,
            # current_act_mask=current_act_mask,
            # activation_mask_orig=activation_mask_orig,
            # sweep_params=sweep_params,
        )


def merge_iteration(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig | MergeRunConfig,
    component_labels: list[str],
    initial_merge: GroupMerge | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    prefix: str = "",
    plot_callback: Callable[..., None] | None = None,
    artifact_callback: Callable[[MergeHistory, int], None] | None = None,
) -> MergeHistory:
    """primary component merging function

    Args:
        activations: (samples, c_components) component causal importances
        merge_config: configuration for the merge process
        component_labels: list of strings naming each component, should be length c_components
        initial_merge: optional initial GroupMerge to start from. if None, starts from identity
        wandb_run: optional wandb run for logging
        prefix: optional string prefix for progress bar, usually used to identify different parallel runs in an ensemble
        plot_callback: optional function to call for plotting at intervals
        artifact_callback: optional function to call to save artifacts at intervals

    """
    # setup
    # ==================================================
    # compute coactivations
    activation_mask_orig: Float[Tensor, "samples c_components"] | None = (
>>>>>>> chinyemba/feature/clustering-sjcs
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
<<<<<<< HEAD
    coact: Float[Tensor, "c c"] = activation_mask_orig.float().T @ activation_mask_orig.float()
=======
    coact: Float[Tensor, "c_components c_components"] = (
        activation_mask_orig.float().T @ activation_mask_orig.float()
    )
>>>>>>> chinyemba/feature/clustering-sjcs

    # check shapes
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"
<<<<<<< HEAD

    # determine number of iterations based on config and number of components
    num_iters: int = merge_config.get_num_iters(c_components)

    # pop logic setup
    # --------------------------------------------------
=======
    assert activation_mask_orig.shape[1] == c_components, (
        "Activation mask must match coactivation matrix shape"
    )

>>>>>>> chinyemba/feature/clustering-sjcs
    # for speed, we precompute whether to pop components and which components to pop
    # if we are not popping, we don't need these variables and can also delete other things
    do_pop: bool = merge_config.pop_component_prob > 0.0
    if do_pop:
        # at each iteration, we will pop a component with probability `pop_component_prob`
        iter_pop: Bool[Tensor, " iters"] = (
<<<<<<< HEAD
            torch.rand(num_iters, device=coact.device) < merge_config.pop_component_prob
=======
            torch.rand(merge_config.iters, device=coact.device) < merge_config.pop_component_prob
>>>>>>> chinyemba/feature/clustering-sjcs
        )
        # we pick a subcomponent at random, and if we decide to pop, we pop that one out of its group
        # if the component is a singleton, nothing happens. this naturally biases towards popping
        # less at the start and more at the end, since the effective probability of popping a component
        # is actually something like `pop_component_prob * (c_components - k_groups) / c_components`
        pop_component_idx: Int[Tensor, " iters"] = torch.randint(
<<<<<<< HEAD
            0, c_components, (num_iters,), device=coact.device
        )

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
=======
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
    iter_idx: int = 0

    # variables we keep track of
    merge_history: MergeHistory = MergeHistory.from_config(
        config=merge_config,
        c_components=c_components,
        labels=component_labels,
        wandb_url=wandb_run.url if wandb_run else None,
>>>>>>> chinyemba/feature/clustering-sjcs
    )

    # free up memory
    if not do_pop:
        del coact
        del activation_mask_orig
<<<<<<< HEAD
=======
        del activations
>>>>>>> chinyemba/feature/clustering-sjcs
        activation_mask_orig = None

    # merge iteration
    # ==================================================
<<<<<<< HEAD
    pbar: tqdm[int] = tqdm(
        range(num_iters),
        unit="iter",
        total=num_iters,
=======
    # while i < merge_config.iters:
    pbar: tqdm[int] = tqdm(
        range(merge_config.iters),
        unit="iter",
        total=merge_config.iters,
>>>>>>> chinyemba/feature/clustering-sjcs
    )
    for iter_idx in pbar:
        # pop components
        # --------------------------------------------------
        if do_pop and iter_pop[iter_idx]:  # pyright: ignore[reportPossiblyUnboundVariable]
            # we split up the group which our chosen component belongs to
            pop_component_idx_i: int = int(pop_component_idx[iter_idx].item())  # pyright: ignore[reportPossiblyUnboundVariable]
            n_components_in_pop_grp: int = int(
                current_merge.components_per_group[  # pyright: ignore[reportArgumentType]
                    current_merge.group_idxs[pop_component_idx_i].item()
                ]
            )

            # but, if the component is the only one in its group, there is nothing to do
            if n_components_in_pop_grp > 1:
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

        # compute costs, figure out what to merge
        # --------------------------------------------------
        # HACK: this is messy
<<<<<<< HEAD
        costs: ClusterCoactivationShaped = compute_merge_costs(
=======
        costs: Float[Tensor, "c_components c_components"] = compute_merge_costs(
>>>>>>> chinyemba/feature/clustering-sjcs
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=merge_config.alpha,
        )

<<<<<<< HEAD
        merge_pair: MergePair = merge_config.merge_pair_sample(costs)

        # merge the pair
        # --------------------------------------------------
        # we do this *before* logging, so we can see how the sampled pair cost compares
=======
        merge_pair: tuple[int, int] = merge_config.merge_pair_sample(costs)

        # handle logging/history/artifacts/progress bar/plotting/etc
        # basically, everything that is not the actual merge computation
        # --------------------------------------------------
        _wandb_iter_log(
            # general
            wandb_run=wandb_run,
            merge_config=merge_config,
            merge_history=merge_history,
            component_labels=component_labels,
            # dims
            iter_idx=iter_idx,
            k_groups=k_groups,
            n_samples=current_act_mask.shape[0],
            # actual data
            current_merge=current_merge,
            merge_pair=merge_pair,
            costs=costs,
            current_coact=current_coact,
            # progress bar stuff
            prefix=prefix,
            pbar=pbar,
            # callbacks
            artifact_callback=artifact_callback,
            plot_function=plot_callback,
        )

        # merge the pair
        # --------------------------------------------------
        # we do this *after* logging, so we can see how the sampled pair cost compares
>>>>>>> chinyemba/feature/clustering-sjcs
        # to the costs of all the other possible pairs
        current_merge, current_coact, current_act_mask = recompute_coacts_merge_pair(
            coact=current_coact,
            merges=current_merge,
            merge_pair=merge_pair,
            activation_mask=current_act_mask,
        )

<<<<<<< HEAD
        # metrics and logging
        # --------------------------------------------------
        # Store in history
        merge_history.add_iteration(
            idx=iter_idx,
            selected_pair=merge_pair,
            current_merge=current_merge,
        )

        # Compute metrics for logging
        # the MDL loss computed here is the *cost of the current merge*, a single scalar value
        # rather than the *delta in cost from merging a specific pair* (which is what `costs` matrix contains)
        diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / current_act_mask.shape[0]
        # this is the cost for the selected pair
        merge_pair_cost: float = float(costs[merge_pair].item())

        # Update progress bar
        pbar.set_description(
            f"{pbar_prefix} k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={merge_pair_cost:.4f}"
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

=======
>>>>>>> chinyemba/feature/clustering-sjcs
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

        # early stopping failsafe
        # --------------------------------------------------
        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {iter_idx} as only {k_groups} groups left",
<<<<<<< HEAD
                stacklevel=2,
=======
                stacklevel=1,
>>>>>>> chinyemba/feature/clustering-sjcs
            )
            break

    # finish up
    # ==================================================
    return merge_history
<<<<<<< HEAD
=======


def merge_iteration_ensemble(
    activations: Float[Tensor, "samples c_components"],
    merge_config: MergeConfig,
    ensemble_size: int,
    component_labels: list[str],
    initial_merge: GroupMerge | None = None,
) -> MergeHistoryEnsemble:
    """Run many merge iterations"""

    output: list[MergeHistory] = []
    for _ in tqdm(range(ensemble_size), unit="ensemble"):
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
>>>>>>> chinyemba/feature/clustering-sjcs
