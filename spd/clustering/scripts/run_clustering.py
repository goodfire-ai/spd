"""Perform a single clustering run.

This can be run as a standalone script, or called via `spd-cluster` (i.e. clustering/scripts/run_pipeline.py).
If the latter, then the following values of ClusteringRunConfig will be overridden by the values
obtained from the submitter:
- dataset_seed
- idx_in_ensemble
- base_output_dir
- ensemble_id

Output structure:
    <base_output_dir>/ (e.g. SPD_CACHE_DIR / "clustering")
    └── <runs>/
        └── <ensemble_id>_<idx_in_ensemble>/  # ensemble_id is randomly generated if not passed in via CLI
            ├── clustering_run_config.json
            ├── history.npz


"""

import argparse
import gc
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor
from wandb.sdk.wandb_run import Run

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.consts import (
    ActivationsTensor,
    BatchTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
)
from spd.clustering.dataset import load_dataset
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import ClusteringRunConfig
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model

# Filenames saved to in this script
CONFIG_FILENAME = "clustering_run_config.json"
HISTORY_FILENAME = "history.npz"


LogCallback = Callable[
    [
        ClusterCoactivationShaped,
        ComponentLabels,
        GroupMerge,
        ClusterCoactivationShaped,
        MergeHistory,
        int,
        int,
        float,
        float,
        float,
        Float[Tensor, " k_groups"],
    ],
    None,
]


def _log_merge_history_plots(run: Run, history: MergeHistory) -> None:
    """Log merge history plots to WandB."""
    fig_cs: Figure = plot_merge_history_cluster_sizes(history=history)
    run.log(
        {"plots/merge_history_cluster_sizes": wandb.Image(fig_cs)},
        step=history.n_iters_current,
    )
    plt.close(fig_cs)


def _save_merge_history_artifact(
    run: Run,
    history_path: Path,
    history: MergeHistory,
) -> None:
    """Save merge history as WandB artifact."""
    artifact: wandb.Artifact = wandb.Artifact(
        name="merge_history",
        type="merge_history",
        description="Merge history",
        metadata={"n_iters_current": history.n_iters_current, "filename": str(history_path)},
    )
    artifact.add_file(str(history_path))
    run.log_artifact(artifact)


def _log_callback(
    run: Run,
    run_config: ClusteringRunConfig,
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
) -> None:
    """Callback for logging during merge iteration."""
    if iter_idx % run_config.logging_intervals.stat == 0:
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

    if iter_idx % run_config.logging_intervals.tensor == 0:
        group_sizes: Int[Tensor, " k_groups"] = current_merge.components_per_group

        tensor_data: dict[str, Tensor] = {
            "coactivation": current_coact,
            "costs": costs,
            "group_sizes": group_sizes,
            "group_activations": diag_acts,
            "group_activations_over_sizes": (
                diag_acts / group_sizes.to(device=diag_acts.device).float()
            ),
        }

        fraction_singleton_groups: float = (group_sizes == 1).float().mean().item()
        if fraction_singleton_groups > 0:
            tensor_data["group_sizes.log1p"] = torch.log1p(group_sizes.float())

        fraction_zero_coacts: float = (current_coact == 0).float().mean().item()
        if fraction_zero_coacts > 0:
            tensor_data["coactivation.log1p"] = torch.log1p(current_coact.float())

        wandb_log_tensor(run, tensor_data, name="iters", step=iter_idx)

        run.log(
            {
                "fraction_singleton_groups": float(fraction_singleton_groups),
                "num_nonsingleton_groups": int((group_sizes > 1).sum().item()),
                "fraction_zero_coacts": float(fraction_zero_coacts),
            },
            step=iter_idx,
        )

    if iter_idx > 0 and iter_idx % run_config.logging_intervals.artifact == 0:
        with tempfile.NamedTemporaryFile() as tmp_file:
            file: Path = Path(tmp_file.name)
            merge_history.save(file)
            artifact: wandb.Artifact = wandb.Artifact(
                name=f"merge_hist_iter.iter_{iter_idx}",
                type="merge_hist_iter",
                description=f"Group indices at iteration {iter_idx}",
                metadata={
                    "iteration": iter_idx,
                    "config": merge_history.merge_config.model_dump(mode="json"),
                },
            )
            artifact.add_file(str(file))
            run.log_artifact(artifact)

    if iter_idx % run_config.logging_intervals.plot == 0:
        fig: Figure = plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            iteration=iter_idx,
            component_labels=component_labels,
            show=False,
        )
        run.log({"plots/merges": wandb.Image(fig)}, step=iter_idx)
        plt.close(fig)


def main(run_config: ClusteringRunConfig) -> Path:
    """A single clustering run.

    Args:
        run_config: Runtime parameters for this clustering run

    Returns:
        Path to saved merge history file
    """
    logger.info("Starting clustering run")
    device = get_device()

    spd_run = SPDRunInfo.from_path(run_config.model_path)
    task_name: TaskName = spd_run.config.task_config.task_name

    # 1. Load dataset
    logger.info(f"Loading dataset (seed={run_config.dataset_seed})")
    load_dataset_kwargs: dict[str, Any] = dict()
    if run_config.dataset_streaming:
        logger.info("Using streaming dataset loading")
        load_dataset_kwargs["config_kwargs"] = dict(streaming=True)
        assert task_name == "lm", (
            f"Streaming dataset loading only supported for 'lm' task, got '{task_name = }'. Remove dataset_streaming=True from config or use a different task."
        )
    batch: BatchTensor = load_dataset(
        model_path=run_config.model_path,
        task_name=task_name,
        batch_size=run_config.batch_size,
        seed=run_config.dataset_seed,
        **load_dataset_kwargs,
    )
    batch = batch.to(device)

    # 2. Setup WandB for this run
    wandb_run: Run | None = None
    if run_config.wandb_project is not None:
        wandb_run = wandb.init(
            entity=run_config.wandb_entity,
            project=run_config.wandb_project,
            group=run_config.ensemble_id,
            config=run_config.model_dump(mode="json"),
            tags=[
                "clustering",
                f"task:{task_name}",
                f"model:{run_config.wandb_decomp_model}",
                f"ensemble_id:{run_config.ensemble_id}",
                f"idx:{run_config.idx_in_ensemble}",
            ],
        )
        logger.info(f"WandB run: {wandb_run.url}")

    # 3. Load model
    logger.info("Loading model")
    model = ComponentModel.from_run_info(spd_run).to(device)

    # 4. Compute activations
    logger.info("Computing activations")
    activations_dict: (
        dict[str, Float[Tensor, "batch seq C"]] | dict[str, Float[Tensor, "batch C"]]
    ) = component_activations(
        model=model,
        batch=batch,
        device=device,
        sigmoid_type=spd_run.config.sigmoid_type,
    )

    # 5. Process activations
    logger.info("Processing activations")
    processed_activations: ProcessedActivations = process_activations(
        activations=activations_dict,
        filter_dead_threshold=run_config.merge_config.filter_dead_threshold,
        seq_mode="concat" if task_name == "lm" else None,
        filter_modules=run_config.merge_config.filter_modules,
    )

    # 6. Log activations (if WandB enabled)
    if wandb_run is not None:
        logger.info("Plotting activations")
        plot_activations(
            processed_activations=processed_activations,
            save_dir=None,  # Don't save to disk, only WandB
            n_samples_max=256,
            wandb_run=wandb_run,
        )
        wandb_log_tensor(
            wandb_run,
            processed_activations.activations,
            "activations",
            0,
            single=True,
        )

    # Clean up memory
    activations: ActivationsTensor = processed_activations.activations
    component_labels: ComponentLabels = ComponentLabels(processed_activations.labels.copy())
    del processed_activations
    del activations_dict
    del model
    del batch
    gc.collect()

    # 7. Run merge iteration
    logger.info("Starting merging")
    log_callback: LogCallback | None = (
        partial(_log_callback, run=wandb_run, run_config=run_config)
        if wandb_run is not None
        else None
    )

    history: MergeHistory = merge_iteration(
        merge_config=run_config.merge_config,
        activations=activations,
        component_labels=component_labels,
        log_callback=log_callback,
    )

    # 8. Save merge history and config
    run_dir = (
        run_config.base_output_dir
        / "runs"
        / f"{run_config.ensemble_id}_{run_config.idx_in_ensemble}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config.to_file(run_dir / CONFIG_FILENAME)
    history_path = run_dir / HISTORY_FILENAME
    history.save(history_path, wandb_url=wandb_run.url if wandb_run else None)
    logger.info(f"✓ History saved to {history_path}")

    # 9. Log to WandB
    if wandb_run is not None:
        _log_merge_history_plots(wandb_run, history)
        _save_merge_history_artifact(wandb_run, history_path, history)
        wandb_run.finish()
        logger.info("WandB run finished")

    return history_path


def cli() -> None:
    """CLI for running a single clustering run."""
    parser = argparse.ArgumentParser(description="Run clustering on a single dataset")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ClusteringRunConfig file",
    )
    parser.add_argument(
        "--idx-in-ensemble",
        type=int,
        default=0,
        help="Index of this run in the ensemble",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=None,
        help="Directory to save merge history",
    )
    parser.add_argument(
        "--ensemble-id",
        type=str,
        default=None,
        help="Ensemble identifier for WandB grouping",
    )
    parser.add_argument(
        "--dataset-streaming",
        action="store_true",
        help="Whether to use streaming dataset loading (if supported by the dataset). see https://github.com/goodfire-ai/spd/pull/199",
    )

    args = parser.parse_args()

    run_config = ClusteringRunConfig.from_file(args.config)

    # Replace values in the run_config from those passed in via CLI (which may come from the
    # pipeline submitter in spd/clustering/scripts/run_pipeline.py)
    overrides: dict[str, Any] = {
        "dataset_seed": run_config.dataset_seed + args.idx_in_ensemble,
        "idx_in_ensemble": args.idx_in_ensemble,
        "dataset_streaming": args.dataset_streaming,
    }
    if args.base_output_dir is not None:
        overrides["base_output_dir"] = args.base_output_dir
    if args.ensemble_id is not None:
        overrides["ensemble_id"] = args.ensemble_id
    run_config = replace_pydantic_model(run_config, overrides)

    main(run_config)


if __name__ == "__main__":
    cli()
