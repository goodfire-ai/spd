import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from muutils.spinner import SpinnerContext
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.merge_run_config import MergeRunConfig
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig, ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import REPO_ROOT
from spd.spd_types import TaskName


def split_dataset_lm(
    model_path: str,
    n_batches: int,
    batch_size: int,
    base_path: Path = REPO_ROOT / "data/split_datasets",
    save_file_fmt: str = "batchsize_{batch_size}/batch_{batch_idx}.npz",
    cfg_file_fmt: str = "batchsize_{batch_size}/_config.json",
) -> tuple[Path, dict[str, Any]]:
    """split up a SS dataset into n_batches of batch_size, returned the saved paths

    1. load the config for a SimpleStories SPD Run given by model_path
    2. create a DataLoader for the dataset
    3. iterate over the DataLoader and save each batch to a file


    """
    with SpinnerContext(message=f"Loading SPD Run Config for '{model_path}'"):
        spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
        cfg: Config = spd_run.config

        try:
            pretrained_model_name: str = cfg.pretrained_model_name  # pyright: ignore[reportAssignmentType]
            assert pretrained_model_name is not None
        except Exception as e:
            raise AttributeError(
                "Could not find 'pretrained_model_name' in the SPD Run config, but called `split_dataset_lm`"
            ) from e

        assert isinstance(cfg.task_config, LMTaskConfig), (
            f"Expected task_config to be of type LMTaskConfig since using `split_dataset_lm`, but got {type(cfg.task_config) = }"
        )

        dataset_config: DatasetConfig = DatasetConfig(
            name=cfg.task_config.dataset_name,
            hf_tokenizer_path=pretrained_model_name,
            split=cfg.task_config.train_data_split,
            n_ctx=cfg.task_config.max_seq_len,
            is_tokenized=False,
            streaming=False,
            seed=0,
            column_name=cfg.task_config.column_name,
        )

    with SpinnerContext(message="getting dataloader..."):
        dataloader: DataLoader[dict[str, torch.Tensor]]
        dataloader, _tokenizer = create_data_loader(
            dataset_config=dataset_config,
            batch_size=batch_size,
            buffer_size=cfg.task_config.buffer_size,
            global_seed=cfg.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

    # make dirs
    base_path.mkdir(parents=True, exist_ok=True)
    (
        base_path
        / save_file_fmt.format(batch_size=batch_size, batch_idx="XX", n_batches=f"{n_batches:02d}")
    ).parent.mkdir(parents=True, exist_ok=True)
    # iterate over the requested number of batches and save them
    output_paths: list[Path] = []
    for batch_idx, batch in tqdm(
        enumerate(iter(dataloader)),
        total=n_batches,
        unit="batch",
    ):
        if batch_idx >= n_batches:
            break
        batch_path: Path = base_path / save_file_fmt.format(
            batch_size=batch_size,
            batch_idx=f"{batch_idx:02d}",
            n_batches=f"{n_batches:02d}",
        )
        np.savez_compressed(
            batch_path,
            input_ids=batch["input_ids"].cpu().numpy(),
        )
        output_paths.append(batch_path)

    # save a config file
    cfg_path: Path = base_path / cfg_file_fmt.format(batch_size=batch_size)
    cfg_data: dict[str, Any] = dict(
        # args to this function
        model_path=model_path,
        batch_size=batch_size,
        n_batches=n_batches,
        # dataset and tokenizer config
        dataset_config=dataset_config.model_dump(mode="json"),
        tokenizer_path=str(getattr(_tokenizer, "name_or_path", None)),
        tokenizer_type=str(getattr(_tokenizer, "__class__", None)),
        # files we saved
        output_files=[str(p) for p in output_paths],
        output_dir=str(base_path),
        output_file_fmt=save_file_fmt,
        cfg_file_fmt=cfg_file_fmt,
        cfg_file=str(cfg_path),
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent="\t"))

    print(f"Saved config to: {cfg_path}")

    return cfg_path, cfg_data


def split_dataset_resid_mlp(
    model_path: str,
    n_batches: int,
    batch_size: int,
    base_path: Path = REPO_ROOT / "data/split_datasets",
    save_file_fmt: str = "batchsize_{batch_size}/batch_{batch_idx}.npz",
    cfg_file_fmt: str = "batchsize_{batch_size}/_config.json",
) -> tuple[Path, dict[str, Any]]:
    """Split a ResidMLP dataset into n_batches of batch_size and save the batches."""
    from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
    from spd.utils.data_utils import DatasetGeneratedDataLoader

    with SpinnerContext(message=f"Loading SPD Run Config for '{model_path}'"):
        spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
        # SPD_RUN = SPDRunInfo.from_path(EXPERIMENT_REGISTRY["resid_mlp3"].canonical_run)
        component_model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
        cfg: Config = spd_run.config

    with SpinnerContext(message="Creating ResidMLPDataset..."):
        assert isinstance(cfg.task_config, ResidMLPTaskConfig), (
            f"Expected task_config to be of type ResidMLPTaskConfig since using `split_dataset_resid_mlp`, but got {type(cfg.task_config) = }"
        )
        assert isinstance(component_model.target_model, ResidMLP), (
            f"Expected patched_model to be of type ResidMLP since using `split_dataset_resid_mlp`, but got {type(component_model.patched_model) = }"
        )

        assert isinstance(component_model.target_model.config, ResidMLPModelConfig), (
            f"Expected patched_model.config to be of type ResidMLPModelConfig since using `split_dataset_resid_mlp`, but got {type(component_model.target_model.config) = }"
        )
        resid_mlp_dataset_kwargs: dict[str, Any] = dict(
            n_features=component_model.target_model.config.n_features,
            feature_probability=cfg.task_config.feature_probability,
            device="cpu",
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=cfg.task_config.data_generation_type,
        )
        dataset: ResidMLPDataset = ResidMLPDataset(**resid_mlp_dataset_kwargs)

        dataloader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # make dirs
    base_path.mkdir(parents=True, exist_ok=True)
    (
        base_path
        / save_file_fmt.format(batch_size=batch_size, batch_idx="XX", n_batches=f"{n_batches:02d}")
    ).parent.mkdir(parents=True, exist_ok=True)

    # iterate over the requested number of batches and save them
    output_paths: list[Path] = []
    batch: torch.Tensor
    # second term in the tuple is same as the first
    for batch_idx, (batch, _) in tqdm(
        enumerate(iter(dataloader)),
        total=n_batches,
        unit="batch",
    ):
        if batch_idx >= n_batches:
            break

        batch_path: Path = base_path / save_file_fmt.format(
            batch_size=batch_size,
            batch_idx=f"{batch_idx:02d}",
            n_batches=f"{n_batches:02d}",
        )
        np.savez_compressed(
            batch_path,
            input_ids=batch.cpu().numpy(),
        )
        output_paths.append(batch_path)

        # save the config file
    cfg_path: Path = base_path / cfg_file_fmt.format(batch_size=batch_size)
    cfg_data: dict[str, Any] = dict(
        # args to this function
        model_path=model_path,
        batch_size=batch_size,
        n_batches=n_batches,
        # dataset and tokenizer config
        resid_mlp_dataset_kwargs=resid_mlp_dataset_kwargs,
        # files we saved
        output_files=[str(p) for p in output_paths],
        output_dir=str(base_path),
        output_file_fmt=save_file_fmt,
        cfg_file_fmt=cfg_file_fmt,
        cfg_file=str(cfg_path),
    )

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent="\t"))
    print(f"Saved config to: {cfg_path}")

    return cfg_path, cfg_data


def split_dataset(
    config: MergeRunConfig | Path,
    base_path: Path = REPO_ROOT / "data/split_datasets",
    save_file_fmt: str = "batchsize_{batch_size}/batch_{batch_idx}.npz",
    cfg_file_fmt: str = "batchsize_{batch_size}/_config.json",
) -> tuple[Path, dict[str, Any]]:
    """Split a dataset into n_batches of batch_size and save the batches"""
    if isinstance(config, Path):
        config = MergeRunConfig.from_file(config)

    model_path: str = config.model_path
    task_name: TaskName = config.task_name
    n_batches: int = config.n_batches
    batch_size: int = config.batch_size

    if task_name == "lm":
        return split_dataset_lm(
            model_path=model_path,
            n_batches=n_batches,
            batch_size=batch_size,
            base_path=base_path,
            save_file_fmt=save_file_fmt,
            cfg_file_fmt=cfg_file_fmt,
        )
    elif task_name == "resid_mlp":
        return split_dataset_resid_mlp(
            model_path=model_path,
            n_batches=n_batches,
            batch_size=batch_size,
            base_path=base_path,
            save_file_fmt=save_file_fmt,
            cfg_file_fmt=cfg_file_fmt,
        )
    else:
        raise ValueError(
            f"Unsupported task name '{task_name}'. Supported tasks are 'lm' and 'resid_mlp'. {model_path=}, {task_name=}"
        )


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="tokenize and split a SimpleStories dataset into smaller batches for clustering ensemble",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to merge run config JSON/YAML file",
    )
    parser.add_argument(
        "--base-path",
        "-p",
        type=Path,
        default=Path("data/split_datasets"),
        help="Base path to save the split datasets to",
    )

    args: argparse.Namespace = parser.parse_args()

    split_dataset(
        config=args.config,
        base_path=args.base_path,
    )
