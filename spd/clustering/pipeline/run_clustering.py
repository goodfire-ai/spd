"""Run clustering on a single dataset (standalone script)."""

import argparse
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float
from matplotlib.figure import Figure
from simple_stories_train.utils import replace_pydantic_model
from torch import Tensor
from wandb.sdk.wandb_run import Run

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import (
    ActivationsTensor,
    BatchTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.pipeline.dataset import load_dataset
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device

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


def run_clustering(run_config: ClusteringRunConfig) -> Path:
    """Run clustering on a single dataset batch.

    Args:
        run_config: Runtime parameters for this clustering run

    Returns:
        Path to saved merge history file
    """
    logger.info("Starting clustering run")
    device = get_device()

    task_name = run_config.get_task_name()

    # 1. Load dataset with run-specific seed
    logger.info(f"Loading dataset (seed={run_config.dataset_seed})")
    batch: BatchTensor = load_dataset(
        model_path=run_config.model_path,
        task_name=task_name,
        batch_size=run_config.batch_size,
        seed=run_config.dataset_seed,
    )
    batch = batch.to(device)
    logger.info(f"Loaded batch with shape {batch.shape}")

    # 2. Setup WandB for this run
    run: Run | None = None
    if run_config.wandb_project is not None:
        run = wandb.init(
            entity=run_config.wandb_entity,
            project=run_config.wandb_project,
            group=run_config.ensemble_id,
            config=run_config.model_dump(mode="json"),
            tags=[
                "clustering",
                f"task:{task_name}",
                f"model:{run_config.wandb_decomp_model}",
                f"idx:{run_config.idx_in_ensemble}",
            ],
        )
        logger.info(f"WandB run: {run.url}")

    # 3. Load model
    logger.info("Loading model")
    spd_run = SPDRunInfo.from_path(run_config.model_path)
    model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path).to(device)
    logger.info("Model loaded")

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
    logger.info("Activations computed")

    # 5. Process activations
    logger.info("Processing activations")
    processed_activations: ProcessedActivations = process_activations(
        activations=activations_dict,
        filter_dead_threshold=run_config.merge_config.filter_dead_threshold,
        seq_mode="concat" if task_name == "lm" else None,
        filter_modules=run_config.merge_config.filter_modules,
    )
    logger.info("Activations processed")

    # 6. Log activations (if WandB enabled)
    if run is not None:
        logger.info("Plotting activations")
        plot_activations(
            processed_activations=processed_activations,
            save_dir=None,  # Don't save to disk, only WandB
            n_samples_max=256,
            wandb_run=run,
        )
        wandb_log_tensor(
            run,
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

    # 7. Run merge iteration
    logger.info("Starting merging")
    log_callback: LogCallback | None = (
        partial(_log_callback, run=run, run_config=run_config) if run is not None else None
    )

    history: MergeHistory = merge_iteration(
        merge_config=run_config.merge_config,
        activations=activations,
        component_labels=component_labels,
        log_callback=log_callback,
    )
    logger.info("Merging complete")

    # 8. Save merge history
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_config.output_dir / f"history_{run_config.idx_in_ensemble}.npz"
    history.save(history_path, wandb_url=run.url if run else None)
    logger.info(f"History saved to {history_path}")

    # 9. Log to WandB
    if run is not None:
        _log_merge_history_plots(run, history)
        _save_merge_history_artifact(run, history_path, history)
        run.finish()
        logger.info("WandB run finished")

    logger.info(f"Clustering complete: {history_path}")
    return history_path


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
    if iter_idx % run_config.intervals["stat"] == 0:
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

    if iter_idx % run_config.intervals["tensor"] == 0:
        from jaxtyping import Int

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
                "fraction_zero_coacts": float(fraction_zero_coacts),
            },
            step=iter_idx,
        )

    if iter_idx > 0 and iter_idx % run_config.intervals["artifact"] == 0:
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

    if iter_idx % run_config.intervals["plot"] == 0:
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
        required=True,
        default=0,
        help="Index of this run in the ensemble",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save merge history",
    )
    parser.add_argument(
        "--ensemble-id",
        type=str,
        required=False,
        default=None,
        help="Ensemble identifier for WandB grouping",
    )

    args = parser.parse_args()

    run_config = ClusteringRunConfig.load(args.config)

    # Replace values in the run_config from those passed in via CLI (which may come from the
    # ensemble submitter)
    run_config = replace_pydantic_model(
        run_config,
        {
            "dataset_seed": run_config.dataset_seed + args.idx_in_ensemble,
            "idx_in_ensemble": args.idx_in_ensemble,
            "output_dir": args.output_dir,
            "ensemble_id": args.ensemble_id,
        },
    )

    history_path = run_clustering(run_config)

    print(f"✓ Merge history saved: {history_path}")


if __name__ == "__main__":
    cli()
