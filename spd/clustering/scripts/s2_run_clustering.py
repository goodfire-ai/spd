# %%

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Int
from muutils.dbg import dbg_auto
from torch import Tensor
from zanj import ZANJ

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.merge_history import MergeHistory
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_history_costs
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import TaskName
from spd.settings import REPO_ROOT

# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false


def run_clustering(
    config: MergeRunConfig | Path,
    dataset_path: Path,
    save_dir: Path = REPO_ROOT / "data/clustering/merge_history/wip/",
    device: str = "cuda",
    plot: bool = True,
) -> Path:
    # Load config
    if isinstance(config, Path):
        config = MergeRunConfig.from_file(config)
    
    model_path: str = config.model_path
    task_name: TaskName = config.task_name

    # get the dataset -- for ensembles, each instance of this script gets a different batch
    data_batch: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(dataset_path)["input_ids"])

    this_merge_path: Path = (
        save_dir
        / f"{config.run_id}-data_{dataset_path.stem}"
    )
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

    dbg_auto(component_acts)
    # process the activations by:
    # 1. filtering out dead components
    # 2. concatenating the activations across the sequence
    # 3. computing coactivations
    processed_activations: dict[str, Any] = process_activations(
        component_acts,
        filter_dead_threshold=config.filter_dead_threshold,
        seq_mode="concat" if config.task_name == "lm" else None,
        filter_modules=config.filter_modules,
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
            # figsize_raw=figsize_raw,
            # figsize_concat=figsize_concat,
            # figsize_coact=figsize_coact,
            # hist_scales=hist_scales,
            # hist_bins=hist_bins,
        )

    # run the merge iteration
    merge_history: MergeHistory = merge_iteration(
        activations=processed_activations["activations"],
        merge_config=config.to_merge_config(),
        component_labels=processed_activations["labels"],
    )

    # save the merge iteration
    hist_save_path: Path = Path(this_merge_path.as_posix() + ".zanj")
    ZANJ().save(merge_history.serialize(), hist_save_path)
    print(f"Merge history saved to {hist_save_path}")

    if plot:
        plot_merge_history_cluster_sizes(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )
        plot_merge_history_costs(
            history=merge_history,
            file_prefix=(this_merge_figs / "merge").as_posix(),
        )

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

    args: argparse.Namespace = parser.parse_args()

    run_clustering(
        config=args.config,
        dataset_path=args.dataset_path,
        device=args.device,
        save_dir=args.save_dir,
    )
