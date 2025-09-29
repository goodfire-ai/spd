from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor
from wandb.sdk.wandb_run import Run

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.compute_costs import compute_mdl_cost
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo


@dataclass
class ClusteringResult:
    history_save_path: Path
    wandb_url: str | None


def process_batches_parallel(
    config: MergeRunConfig,
    data_files: list[Path],
    output_dir: Path,
    workers_per_device: int,
    devices: list[str],
) -> list[ClusteringResult]:
    worker_args = [
        (config, data_path, output_dir, devices[i % len(devices)])
        for i, data_path in enumerate(data_files)
    ]

    with Pool(workers_per_device * len(devices)) as pool:
        results = pool.map(_worker_fn, worker_args)

    return results


def _worker_fn(args: tuple[MergeRunConfig, Path, Path, str]) -> ClusteringResult:
    return _run_clustering(*args)


def _run_clustering(
    config: MergeRunConfig,
    data_path: Path,
    output_base_dir: Path,
    device: str,
) -> ClusteringResult:
    batch_id = data_path.stem

    run = _setup_wandb(batch_id=batch_id, config=config) if config.wandb_enabled else None

    this_merge_dir = output_base_dir / f"data_{batch_id}"
    this_merge_plots_dir = this_merge_dir / "plots"

    spd_run = SPDRunInfo.from_path(config.model_path)
    model = ComponentModel.from_pretrained(spd_run.checkpoint_path).to(device)

    batch = _load_batch_data(data_path).to(device)

    compoenent_activations = component_activations(
        model=model,
        batch=batch,
        device=device,
        sigmoid_type=spd_run.config.sigmoid_type,
    )

    processed_activations = process_activations(
        activations=compoenent_activations,
        filter_dead_threshold=config.filter_dead_threshold,
        seq_mode="concat" if config.task_name == "lm" else None,
        filter_modules=config.filter_modules,
    )

    if run is not None:
        wandb_log_tensor(
            run=run,
            data=processed_activations.activations,
            name="processed_activations",
            step=0,
            single=True,
        )
        wandb_url = run.url
    else:
        wandb_url = None

    # Use original activations for raw plots, but filtered data for concat/coact/histograms
    logger.info("plotting")
    plot_activations(
        processed_activations=processed_activations,
        save_dir=this_merge_plots_dir,
        wandb_run=run,
    )

    logger.info("cleaning up memory")
    activations = processed_activations.activations
    component_labels = processed_activations.labels.copy()
    del processed_activations  # we copied what we needed
    del compoenent_activations  # processed already
    del model  # already did the forward pass
    del batch  # already did the forward pass

    log_callback = (
        partial(_log_callback, run=run, batch_id=batch_id, config=config)
        if run is not None
        else None
    )

    history = merge_iteration(config, batch_id, activations, component_labels, log_callback)

    history_save_path = this_merge_dir / "merge_history.zip"

    history.save(history_save_path, wandb_url=wandb_url)

    if run is not None:
        _log_merge_history_plots_to_wandb(run, history)
        _save_merge_history_to_wandb(
            run, history_save_path, batch_id, config.config_identifier, history
        )

        run.finish()

    return ClusteringResult(history_save_path=history_save_path, wandb_url=wandb_url)


def _load_batch_data(data_path: Path) -> Int[Tensor, "batch_size n_ctx"]:
    """Load a batch of data from disk."""
    data = np.load(data_path)
    return torch.tensor(data["input_ids"])


def _setup_wandb(
    batch_id: str,
    config: MergeRunConfig,
) -> Run:
    run = wandb.init(
        project=config.wandb_project,
        name=f"{config.config_identifier}-{batch_id}",
        group=config.wandb_group,
        config=config.model_dump_with_properties(),
        tags=[
            "cluster-run",
            f"model:{config.wandb_decomp_model}",
            f"task:{config.task_name}",
            f"batch:{batch_id}",
            f"config:{config.config_identifier}",
        ],
    )
    logger.info(f"Initialized WandB run: {run.name} in group {config.wandb_group}")
    return run


def _log_merge_history_plots_to_wandb(run: Run, history: MergeHistory):
    fig_cs = plot_merge_history_cluster_sizes(history=history)
    run.log(
        {"plots/merge_history_cluster_sizes": wandb.Image(fig_cs)},
        step=history.n_iters_current,
    )
    plt.close(fig_cs)


def _save_merge_history_to_wandb(
    run: Run,
    history_path: Path,
    batch_id: str,
    config_identifier: str,
    history: MergeHistory,
):
    artifact = wandb.Artifact(
        name=f"merge_history_{batch_id}",
        type="merge_history",
        description=f"Merge history for batch {batch_id}",
        metadata={
            "batch_name": batch_id,
            "config_identifier": config_identifier,
            "n_iters_current": history.n_iters_current,
            "filename": history_path,
        },
    )
    artifact.add_file(str(history_path))
    run.log_artifact(artifact)


def _log_callback(
    run: Run,
    batch_id: str,
    current_coact: Float[Tensor, "k_groups k_groups"],
    component_labels: list[str],
    current_merge: GroupMerge,
    config: MergeRunConfig,
    costs: Float[Tensor, "k_groups k_groups"],
    merge_history: MergeHistory,
    iter_idx: int,
    k_groups: int,
    merge_pair_cost: float,
    mdl_loss: float,
    mdl_loss_norm: float,
    diag_acts: Float[Tensor, " k_groups"],
):
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
        with tempfile.NamedTemporaryFile() as tmp_file:
            file = Path(tmp_file.name)
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
