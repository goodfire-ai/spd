# %%

import functools
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import wandb
import wandb.sdk.wandb_run
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from muutils.dbg import dbg_auto
from torch import Tensor
from zanj import ZANJ

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.merge import (
    plot_merge_history_cluster_sizes,
    plot_merge_history_costs,
    plot_merge_iteration,
)
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import REPO_ROOT

# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false

# Global batch_id for logging
_BATCH_ID: str = "unk"

# Delimiter for structured output parsing
RESULT_DELIMITER: str = "-" * 50


def _open_json_fd() -> TextIO:
    """Open file descriptor for JSON output from environment variable"""
    fd_num: int = int(os.environ["JSON_FD"])
    return os.fdopen(fd_num, "w", buffering=1)


def emit_result(obj: dict[str, str | None]) -> None:
    """Emit result JSON via environment fd"""
    out: TextIO = _open_json_fd()
    print(json.dumps(obj, separators=(",", ":")), file=out, flush=True)


def log(message: str) -> None:
    """Print a message with orange batch ID prefix.

    Works with both regular print and tqdm progress bars.
    """
    # ANSI color codes: \033[38;5;208m is orange, \033[0m resets
    print(f"\033[38;5;208m[{_BATCH_ID}]\033[0m {message}")


def save_group_idxs_artifact(
    merge_hist: MergeHistory,
    iteration: int,
    wandb_run: wandb.sdk.wandb_run.Run,
    save_dir: Path,
    dataset_stem: str,
) -> None:
    """Save merge_hist to file and upload as WandB artifact"""
    # Save to file in the same directory as merge history
    group_idxs_path: Path = save_dir / f"iter_{iteration:04}.zanj"
    merge_hist.save(group_idxs_path)

    # Create and upload artifact
    artifact = wandb.Artifact(
        name=f"merge_hist_iter.{dataset_stem}.iter_{iteration}",
        type="merge_hist_iter",
        description=f"Group indices for batch {dataset_stem} at iteration {iteration}",
        metadata={
            "batch_name": dataset_stem,
            "iteration": iteration,
            "config": merge_hist.config.model_dump(mode="json"),
        },
    )
    artifact.add_file(str(group_idxs_path))
    wandb_run.log_artifact(artifact)


def plot_merge_iteration_callback(
    costs: torch.Tensor,
    merge_history: MergeHistory,
    current_merge: Any,
    current_coact: torch.Tensor,
    i: int,
    component_labels: list[str],
    wandb_run: wandb.sdk.wandb_run.Run | None,
    artifact_frequency: int,
    **kwargs: Any,
) -> None:
    """Plot merge iteration at artifact frequency and log to WandB."""
    assert kwargs  # Ensure unused kwargs are passed

    # Only plot at artifact frequency (same as when we save artifacts)
    if wandb_run is not None and i > 0 and i % artifact_frequency == 0:
        # Create the plot and get the figure
        fig = plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            pair_cost=merge_history.latest()["costs_stats"]["chosen_pair"],  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
            iteration=i,
            component_labels=component_labels,
            show=False,  # Don't display the plot
        )

        # Log to WandB
        wandb_run.log({"plots/merges": wandb.Image(fig)}, step=i)
        plt.close(fig)  # Close figure to free memory


def run_clustering(
    config: MergeRunConfig | Path,
    dataset_path: Path,
    save_dir: Path = REPO_ROOT / "data/clustering/merge_history/wip/",
    device: str = "cuda",
    plot: bool = True,
    sort_components: bool = False,
) -> Path:
    # setup
    # ======================================================================
    # Load config
    config_: MergeRunConfig
    if isinstance(config, Path):  # noqa: SIM108
        config_ = MergeRunConfig.from_file(config)
    else:
        config_ = config

    model_path: str = config_.model_path

    # Extract batch ID from dataset filename (e.g., "batch_01.npz" -> "01")
    global _BATCH_ID
    _BATCH_ID = dataset_path.stem.split("_")[-1] if "_" in dataset_path.stem else dataset_path.stem  # pyright: ignore[reportConstantRedefinition]

    log(f"Starting clustering for {dataset_path.name}")

    # Initialize WandB run if enabled
    wandb_run: wandb.sdk.wandb_run.Run | None = None
    if config_.wandb_enabled:
        wandb_run = wandb.init(
            project=config_.wandb_project,
            name=f"{config_.config_identifier}-{dataset_path.stem}",
            group=config_.wandb_group,
            config=config_.model_dump_with_properties(),
            tags=[
                "cluster-run",
                f"model:{config_.wandb_decomp_model}",
                f"task:{config_.task_name}",
                f"batch:{dataset_path.stem}",
                f"config:{config_.config_identifier}",
            ],
        )
        log(f"Initialized WandB run: {wandb_run.name} in group {config_.wandb_group}")

    this_merge_path: Path = save_dir / f"{config_.config_identifier}-data_{dataset_path.stem}"
    this_merge_figs: Path = this_merge_path / "plots"
    if plot:
        this_merge_figs.mkdir(parents=True, exist_ok=True)

    # Create callbacks if wandb is enabled
    artifact_callback: Callable[[MergeHistory, int], None] | None = None
    plot_callback: Callable[..., None] | None = None

    if wandb_run is not None:
        artifact_callback = functools.partial(
            save_group_idxs_artifact,
            wandb_run=wandb_run,
            save_dir=this_merge_path / "checkpoints",
            dataset_stem=dataset_path.stem,
        )

        plot_callback = functools.partial(
            plot_merge_iteration_callback,
            wandb_run=wandb_run,
            artifact_frequency=config_.wandb_artifact_frequency,
            batch_id=_BATCH_ID,
        )

    # get model and data
    # ======================================================================
    log(f"getting data batch from {dataset_path}")
    # get the dataset -- for ensembles, each instance of this script gets a different batch
    data_batch: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(dataset_path)["input_ids"])

    # load the spd run of the actual model we are decomposing
    log(f"getting spd run {model_path}")
    spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
    component_model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
    component_model.to(device)

    # get, process, and plot component activations
    # ======================================================================
    log("computing activations")
    component_acts: dict[str, Tensor] = component_activations(
        model=component_model,
        batch=data_batch,
        device=device,
        # threshold=0.1,
        # TODO: where can we find this in the model itself???
        sigmoid_type="hard",
    )

    if wandb_run is not None:
        wandb_log_tensor(wandb_run, component_acts, "component_activations", step=0, single=True)
    else:
        dbg_auto(component_acts)

    # process the activations by:
    # 1. filtering out dead components
    # 2. concatenating the activations across the sequence
    # 3. computing coactivations
    log("processing activations")
    processed_activations: ProcessedActivations = process_activations(
        component_acts,
        filter_dead_threshold=config_.filter_dead_threshold,
        seq_mode="concat" if config_.task_name == "lm" else None,
        filter_modules=config_.filter_modules,
        sort_components=sort_components,
    )

    if plot:
        log("plotting")
        # Import plotting function only when needed
        from spd.clustering.plotting.activations import plot_activations

        # Use original activations for raw plots, but filtered data for concat/coact/histograms
        plot_activations(
            processed_activations=processed_activations,
            n_samples_max=256,
            save_pdf=True,
            pdf_prefix=(this_merge_figs / "activations").as_posix(),
            wandb_run=wandb_run,
            log=log,
        )

    # memory cleanup
    # ======================================================================
    # copy what we need, delete the rest to free memory
    log("cleaning up memory")
    activations_: Float[Tensor, "n_steps c"] = processed_activations.activations
    labels: list[str] = processed_activations.labels.copy()
    del processed_activations  # we copied what we needed
    del component_acts  # processed already
    del component_model  # already did the forward pass
    del data_batch  # already did the forward pass

    # run the merge iteration
    # ======================================================================
    log("starting merge iteration")
    merge_history: MergeHistory = merge_iteration(
        activations=activations_,
        merge_config=config_,  # Pass full MergeRunConfig to access wandb_log_frequency
        component_labels=labels,
        wandb_run=wandb_run,
        prefix=f"\033[38;5;208m[{_BATCH_ID}]\033[0m",
        artifact_callback=artifact_callback,
        plot_function=plot_callback,
    )

    # saving and plotting
    # ======================================================================

    # save the merge iteration
    hist_save_path: Path = this_merge_path / "merge_history.zanj"

    merge_history_serialized: dict[str, Any] = merge_history.serialize()
    # TODO: Consider adding fallback to dbg_auto if wandb_run is None
    # For now we skip logging merge_history_serialized as it's large and complex
    # dbg_auto(merge_history_serialized)

    ZANJ().save(merge_history_serialized, hist_save_path)
    log(f"Merge history saved to {hist_save_path}")

    # Save WandB URL to file
    wburl_path: Path | None = None
    wandb_url: str | None = None
    if wandb_run is not None:
        wburl_path = hist_save_path.with_suffix(".wburl")
        if wandb_run.url:
            wburl_path.write_text(wandb_run.url)
            wandb_url = wandb_run.url

    # Save merge history as WandB artifact
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"merge_history_{dataset_path.stem}",
            type="merge_history",
            description=f"Merge history for batch {dataset_path.stem}",
            metadata={
                "batch_name": dataset_path.stem,
                "config_identifier": config_.config_identifier,
                "n_iters_current": merge_history.n_iters_current,
                "filename": hist_save_path,
            },
        )
        # Add both files before logging the artifact
        artifact.add_file(str(hist_save_path))
        wandb_run.log_artifact(artifact)

    if plot:
        fig_cs: plt.Figure = plot_merge_history_cluster_sizes(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )
        fig_costs: plt.Figure = plot_merge_history_costs(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )
        if wandb_run is not None:
            wandb_run.log(
                {"plots/merge_history_cluster_sizes": wandb.Image(fig_cs)},
                step=merge_history.n_iters_current,
            )
            wandb_run.log(
                {"plots/merge_history_costs": wandb.Image(fig_costs)},
                step=merge_history.n_iters_current,
            )
        # Close figures to free memory
        plt.close(fig_cs)
        plt.close(fig_costs)

    # Finish WandB run
    if wandb_run is not None:
        wandb_run.finish()
        log(f"Finished WandB run with url: {wandb_run.url}")

    # Output structured result for main.py to parse via fd 3
    result: dict[str, str | None] = {
        "hist_save_path": str(hist_save_path),
        "wburl_path": str(wburl_path) if wburl_path else None,
        "wandb_url": wandb_url,
        "batch_name": dataset_path.stem,
        "config_identifier": config_.config_identifier,
    }

    emit_result(result)

    return hist_save_path


def cli():
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run a merge iteration on a batch of data using a component model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the merge run config JSON/YAML file",
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=Path,
        required=True,
        help="Path to the dataset file (e.g., a .npz file with input_ids)",
    )
    parser.add_argument(
        "--device",
        "-D",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        type=Path,
        required=True,
        help="Directory to save the merge history",
    )
    # parser.add_argument(
    #     "--sort-components",
    #     action="store_true",
    #     help="Sort components by similarity within each module before concatenation",
    # )
    parser.add_argument(
        "--plot",
        action="store_true",
        dest="plot",
        help="plotting of activations",
    )

    args: argparse.Namespace = parser.parse_args()

    run_clustering(
        config=args.config,
        dataset_path=args.dataset_path,
        device=args.device,
        save_dir=args.save_dir,
        # sort_components=args.sort_components,
        sort_components=True,
        plot=args.plot,
    )


if __name__ == "__main__":
    cli()
