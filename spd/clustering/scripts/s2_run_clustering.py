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
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import REPO_ROOT

# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false


def run_clustering(
    merge_config: MergeConfig | Path,
    dataset_path: Path,
    model_path: str,
    save_dir: Path = REPO_ROOT / "data/clustering/merge_history/wip/",
    device: str = "cuda",
) -> Path:
    # get the merge config
    merge_config_: MergeConfig
    if isinstance(merge_config, Path):
        merge_config_ = MergeConfig.model_validate_json(merge_config.read_text())
    elif isinstance(merge_config, MergeConfig):
        merge_config_ = merge_config
    else:
        raise TypeError("merge_config must be a MergeConfig or a Path to a JSON file")

    # get the dataset -- for ensembles, each instance of this script gets a different batch
    data_batch: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(dataset_path)["input_ids"])

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
        filter_dead_threshold=merge_config_.filter_dead_threshold,
        seq_mode="concat",
        plots=True,  # Plot the processed activations
        # plot_title="Processed Activations",
    )

    # run the merge iteration
    merge_history: MergeHistory = merge_iteration(
        activations=processed_activations["activations"],
        merge_config=merge_config_,
        component_labels=processed_activations["labels"],
    )

    # save the merge iteration
    save_path: Path = (
        save_dir
        / f"history_{merge_config_.stable_hash}_{data_batch.shape[0]}_{dataset_path.stem}.zanj"
    )
    ZANJ().save(merge_history.serialize(), save_path)
    print(f"Merge history saved to {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run a merge iteration on a batch of data using a component model."
    )
    parser.add_argument(
        "--merge-config",
        "-c",
        type=Path,
        required=True,
        help="Path to the merge config JSON file",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to the SPD Run, usually a wandb run",
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
        merge_config=args.merge_config,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=args.device,
        save_dir=args.save_dir,
    )
