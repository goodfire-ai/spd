"""
Merge iteration with logging support.

This wraps the pure merge_iteration_pure() function and adds WandB/plotting callbacks.
"""

import tempfile
import warnings
from pathlib import Path

import torch
import wandb
import wandb.sdk.wandb_run
from jaxtyping import Bool, Float, Int
from pandas.io.formats.style import plt
from torch import Tensor
from tqdm import tqdm

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_pop_group,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.merge import plot_merge_iteration
from spd.clustering.wandb_tensor_info import wandb_log_tensor


def merge_iteration(
    config: MergeRunConfig,
    batch_id: str,
    activations: Float[Tensor, "n_steps c"],
    component_labels: list[str],
    run: wandb.sdk.wandb_run.Run | None = None,
) -> MergeHistory:
    """
    Merge iteration with optional logging/plotting callbacks.

    This wraps the pure computation with logging capabilities while maintaining
    the same core algorithm logic.
    """

    # Compute coactivations
    activation_mask_orig = (
        activations > config.activation_threshold
        if config.activation_threshold is not None
        else activations
    )
    coact = activation_mask_orig.float().T @ activation_mask_orig.float()

    # Setup
    c_components = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"

    # Prepare pop component logic
    do_pop = config.pop_component_prob > 0.0
    if do_pop:
        iter_pop = torch.rand(config.iters, device=coact.device) < config.pop_component_prob
        pop_component_idx = torch.randint(0, c_components, (config.iters,), device=coact.device)

    # for speed, we precompute whether to pop components and which components to pop
    # if we are not popping, we don't need these variables and can also delete other things
    do_pop: bool = config.pop_component_prob > 0.0
    if do_pop:
        # at each iteration, we will pop a component with probability `pop_component_prob`
        iter_pop: Bool[Tensor, " iters"] = (
            torch.rand(config.iters, device=coact.device) < config.pop_component_prob
        )
        # we pick a subcomponent at random, and if we decide to pop, we pop that one out of its group
        # if the component is a singleton, nothing happens. this naturally biases towards popping
        # less at the start and more at the end, since the effective probability of popping a component
        # is actually something like `pop_component_prob * (c_components - k_groups) / c_components`
        pop_component_idx: Int[Tensor, " iters"] = torch.randint(
            0, c_components, (config.iters,), device=coact.device
        )

    # Initialize merge
    current_merge = GroupMerge.identity(n_components=c_components)

    # Initialize variables
    k_groups = c_components
    current_coact: Float[Tensor, "k_groups k_groups"] = coact.clone()
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask_orig.clone()

    # Initialize history
    merge_history = MergeHistory.from_config(
        config=config,
        c_components=c_components,
        labels=component_labels,
        wandb_url=run.url if run else None,
    )

    # Memory cleanup
    if not do_pop:
        del coact
        del activation_mask_orig
        activation_mask_orig = None

    # Main iteration loop with progress bar
    pbar = tqdm(range(config.iters), unit="iter", total=config.iters)
    for iter_idx in pbar:
        if do_pop and iter_pop[iter_idx]:  # pyright: ignore[reportPossiblyUnboundVariable]
            assert activation_mask_orig is not None, "Activation mask original is None"

            pop_component_idx_i = int(pop_component_idx[iter_idx].item())  # pyright: ignore[reportPossiblyUnboundVariable]
            group_idx = int(current_merge.group_idxs[pop_component_idx_i].item())
            n_components_in_pop_grp = int(current_merge.components_per_group[group_idx].item())

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
        costs = compute_merge_costs(
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=config.alpha,
        )

        merge_pair = config.merge_pair_sample(costs)

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

        if run:
            _wandb_iter_log(
                run=run,
                batch_id=batch_id,
                current_coact=current_coact,
                component_labels=component_labels,
                current_merge=current_merge,
                config=config,
                current_act_mask=current_act_mask,
                costs=costs,
                merge_pair=merge_pair,
                merge_history=merge_history,
                iter_idx=iter_idx,
                k_groups=k_groups,
                pbar=pbar,
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


def _wandb_iter_log(
    run: wandb.sdk.wandb_run.Run,
    batch_id: str,
    current_coact: Float[Tensor, "k_groups k_groups"],
    component_labels: list[str],
    current_merge: GroupMerge,
    config: MergeRunConfig,
    current_act_mask: Bool[Tensor, "samples k_groups"],
    costs: Float[Tensor, "k_groups k_groups"],
    merge_pair: tuple[int, int],
    merge_history: MergeHistory,
    iter_idx: int,
    k_groups: int,
    pbar: tqdm[int],
):
    # Compute metrics for logging
    diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
    mdl_loss = compute_mdl_cost(
        acts=diag_acts,
        merges=current_merge,
        alpha=config.alpha,
    )
    mdl_loss_norm = mdl_loss / current_act_mask.shape[0]
    merge_pair_cost = float(costs[merge_pair].item())

    # Update progress bar

    prefix = f"\033[38;5;208m[{batch_id}]\033[0m"
    pbar.set_description(
        f"{prefix} k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={merge_pair_cost:.4f}"
    )

    if iter_idx % config.intervals["stat"] == 0:
        run.log(
            {
                "k_groups": int(k_groups),
                "merge_pair_cost": merge_pair_cost,
                "merge_pair_cost_semilog[1e-3]": semilog(merge_pair_cost, epsilon=1e-3),
                "mdl_loss": float(mdl_loss),
                "mdl_loss_norm": float(mdl_loss_norm),
            },
            step=iter_idx,
        )

    if iter_idx % config.intervals["tensor"] == 0:
        group_sizes: Int[Tensor, " k_groups"] = current_merge.components_per_group

        tensor_data = {
            "coactivation": current_coact,
            "costs": costs,
            "group_sizes": group_sizes,
            "group_activations": diag_acts,
            "group_activations_over_sizes": diag_acts
            / group_sizes.to(device=diag_acts.device).float(),
        }

        fraction_singleton_groups = (group_sizes == 1).float().mean().item()
        if fraction_singleton_groups > 0:
            tensor_data["group_sizes.log1p"] = torch.log1p(group_sizes.float())

        fraction_zero_coacts = (current_coact == 0).float().mean().item()
        if fraction_zero_coacts > 0:
            tensor_data["coactivation.log1p"] = torch.log1p(current_coact.float())

        wandb_log_tensor(run, tensor_data, name="iters", step=iter_idx)

        run.log(
            {
                "fraction_singleton_groups": float(fraction_singleton_groups),
                "fraction_zero_coacts": float(fraction_zero_coacts),
            },
            step=iter_idx,
        )

    if iter_idx > 0 and iter_idx % config.intervals["artifact"] == 0:
        with tempfile.TemporaryFile() as tmp_file:
            file: Path = Path(tmp_file.name)
            file.parent.mkdir(parents=True, exist_ok=True)
            merge_history.save(file)
            artifact = wandb.Artifact(
                name=f"merge_hist_iter.{batch_id}.iter_{iter_idx}",
                type="merge_hist_iter",
                description=f"Group indices for batch {batch_id} at iteration {iter_idx}",
                metadata={
                    "batch_name": batch_id,
                    "iteration": iter_idx,
                    "config": merge_history.config.model_dump(mode="json"),
                    "config_identifier": merge_history.config,
                },
            )
            artifact.add_file(str(file))
            run.log_artifact(artifact)

    if iter_idx % config.intervals["plot"] == 0:
        fig = plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            iteration=iter_idx,
            component_labels=component_labels,
            show=False,
        )
        run.log({"plots/merges": wandb.Image(fig)}, step=iter_idx)
        plt.close(fig)
