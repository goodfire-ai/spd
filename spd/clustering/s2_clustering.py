from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Int
from torch import Tensor
from wandb.sdk.wandb_run import Run

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo


@dataclass
class ClusteringResult:
    history_save_path: Path
    wandb_url: str | None


# TODO consider making this a generator
def process_batches_parallel(
    config: MergeRunConfig,
    data_files: list[Path],
    output_base_dir: Path,
    n_workers: int,
    devices: list[str],
) -> list[ClusteringResult]:
    devices = devices or ["cuda:0"]

    # Create worker arguments with device assignment
    worker_args = [
        (config, data_path, output_base_dir, devices[i % len(devices)])
        for i, data_path in enumerate(data_files)
    ]

    with Pool(n_workers) as pool:
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

    history = merge_iteration(config, batch_id, activations, component_labels, run)

    history_save_path = this_merge_dir / "merge_history.zip"

    history.save(history_save_path)

    if run is not None:
        _log_merge_history_plots_to_wandb(run, history)
        # _save_merge_history_to_wandb(
        #     run, history_save_path, batch_id, config.config_identifier, history
        # )

        wandb_url = run.url
        run.finish()
    else:
        wandb_url = None

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
        {
            "plots/merge_history_cluster_sizes": wandb.Image(fig_cs),
        },
        step=history.n_iters_current,
    )

    plt.close(fig_cs)
