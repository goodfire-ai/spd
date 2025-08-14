# %%

from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from jaxtyping import Int
from torch import Tensor
from zanj import ZANJ

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_history_costs
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import REPO_ROOT
from spd.utils.wandb_tensor_info import wandb_log_tensor

# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false


def run_clustering(
    config: MergeRunConfig | Path,
    dataset_path: Path,
    save_dir: Path = REPO_ROOT / "data/clustering/merge_history/wip/",
    device: str = "cuda",
    plot: bool = True,
    sort_components: bool = False,
) -> Path:
    # Load config
    if isinstance(config, Path):
        config = MergeRunConfig.from_file(config)

    model_path: str = config.model_path

    # Initialize WandB run if enabled
    wandb_run: wandb.sdk.wandb_run.Run | None = None
    if config.wandb_enabled:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=f"{config.config_identifier}-{dataset_path.stem}",
            group=config.wandb_group,
            config=config.model_dump_with_properties(),
            tags=[
                f"model:{config.wandb_decomp_model}",
                f"task:{config.task_name}",
                f"batch:{dataset_path.stem}",
                f"config:{config.config_identifier}",
            ],
        )
        logger.info(f"Initialized WandB run: {wandb_run.name} in group {config.wandb_group}")

    # get the dataset -- for ensembles, each instance of this script gets a different batch
    data_batch: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(dataset_path)["input_ids"])

    this_merge_path: Path = save_dir / f"{config.config_identifier}-data_{dataset_path.stem}"
    this_merge_figs: Path = Path(this_merge_path.as_posix() + "_plots/")
    if plot:
        this_merge_figs.mkdir(parents=True, exist_ok=True)

    # load the spd run
    spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
    component_model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
    component_model.to(device)

    # compute the activations for the components
    component_acts: dict[str, Tensor] = component_activations(
        model=component_model,
        batch=data_batch,
        device=device,
        # threshold=0.1,
        # TODO: where can we find this in the model itself???
        sigmoid_type="hard",
    )

    # TODO: Consider adding fallback to dbg_tensor if wandb_run is None
    wandb_log_tensor(wandb_run, component_acts, "component_activations", step=0, single=True)
    # process the activations by:
    # 1. filtering out dead components
    # 2. concatenating the activations across the sequence
    # 3. computing coactivations
    processed_activations: dict[str, Any] = process_activations(
        component_acts,
        filter_dead_threshold=config.filter_dead_threshold,
        seq_mode="concat" if config.task_name == "lm" else None,
        filter_modules=config.filter_modules,
        sort_components=sort_components,
    )

    if plot:
        # Import plotting function only when needed
        from spd.clustering.plotting.activations import plot_activations

        # Use original activations for raw plots, but filtered data for concat/coact/histograms
        plot_activations(
            activations=processed_activations["activations_raw"],
            act_concat=processed_activations["activations"],
            coact=processed_activations["coactivations"],
            labels=processed_activations["labels"],
            save_pdf=True,
            pdf_prefix=(this_merge_figs / "activations").as_posix(),
            wandb_run=wandb_run,
        )

    # run the merge iteration
    merge_history: MergeHistory = merge_iteration(
        activations=processed_activations["activations"],
        merge_config=config,  # Pass full MergeRunConfig to access wandb_log_frequency
        component_labels=processed_activations["labels"],
        wandb_run=wandb_run,
    )

    # save the merge iteration
    hist_save_path: Path = Path(this_merge_path.as_posix() + ".zanj")

    merge_history_serialized: dict[str, Any] = merge_history.serialize()
    # TODO: Consider adding fallback to dbg_auto if wandb_run is None
    # For now we skip logging merge_history_serialized as it's large and complex
    # dbg_auto(merge_history_serialized)

    ZANJ().save(merge_history_serialized, hist_save_path)
    logger.info(f"Merge history saved to {hist_save_path}")

    # Save merge history as WandB artifact
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"merge_history_{dataset_path.stem}",
            type="merge_history",
            description=f"Merge history for batch {dataset_path.stem}",
            metadata={
                "batch_name": dataset_path.stem,
                "config_identifier": config.config_identifier,
                "n_iters_current": merge_history.n_iters_current,
            },
        )
        artifact.add_file(str(hist_save_path))
        wandb_run.log_artifact(artifact)

    if plot:
        plot_merge_history_cluster_sizes(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )
        plot_merge_history_costs(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )

    # Finish WandB run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info(f"Finished WandB run with url: {wandb_run.url}")

    return hist_save_path


if __name__ == "__main__":
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
    parser.add_argument(
        "--sort-components",
        action="store_true",
        help="Sort components by similarity within each module before concatenation",
    )

    args: argparse.Namespace = parser.parse_args()

    run_clustering(
        config=args.config,
        dataset_path=args.dataset_path,
        device=args.device,
        save_dir=args.save_dir,
        sort_components=args.sort_components,
    )
