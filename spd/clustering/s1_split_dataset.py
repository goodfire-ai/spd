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


def split_dataset_lm(
    model_path: str,
    n_batches: int,
    batch_size: int,
    output_dir: Path,
    save_file_fmt: str,
    cfg_file_fmt: str,
) -> list[Path]:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        output_dir
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
        batch_path: Path = output_dir / save_file_fmt.format(
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
    cfg_path: Path = output_dir / cfg_file_fmt.format(batch_size=batch_size)
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
        output_dir=str(output_dir),
        output_file_fmt=save_file_fmt,
        cfg_file_fmt=cfg_file_fmt,
        cfg_file=str(cfg_path),
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent="\t"))

    print(f"Saved config to: {cfg_path}")

    return output_paths


def split_dataset_resid_mlp(
    model_path: str,
    n_batches: int,
    batch_size: int,
    output_dir: Path,
    save_file_fmt: str,
    cfg_file_fmt: str,
) -> list[Path]:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        output_dir
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

        batch_path: Path = output_dir / save_file_fmt.format(
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
    cfg_path: Path = output_dir / cfg_file_fmt.format(batch_size=batch_size)
    cfg_data: dict[str, Any] = dict(
        # args to this function
        model_path=model_path,
        batch_size=batch_size,
        n_batches=n_batches,
        # dataset and tokenizer config
        resid_mlp_dataset_kwargs=resid_mlp_dataset_kwargs,
        # files we saved
        output_files=[str(p) for p in output_paths],
        output_dir=str(output_dir),
        output_file_fmt=save_file_fmt,
        cfg_file_fmt=cfg_file_fmt,
        cfg_file=str(cfg_path),
    )

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent="\t"))
    print(f"Saved config to: {cfg_path}")

    return output_paths


def split_and_save_dataset(
    config: MergeRunConfig,
    output_dir: Path,
    save_file_fmt: str,
    cfg_file_fmt: str,
) -> list[Path]:
    """Split a dataset into n_batches of batch_size and save the batches"""
    match config.task_name:
        case "lm":
            return split_dataset_lm(
                model_path=config.model_path,
                n_batches=config.n_batches,
                batch_size=config.batch_size,
                output_dir=output_dir,
                save_file_fmt=save_file_fmt,
                cfg_file_fmt=cfg_file_fmt,
            )
        case "resid_mlp":
            return split_dataset_resid_mlp(
                model_path=config.model_path,
                n_batches=config.n_batches,
                batch_size=config.batch_size,
                output_dir=output_dir,
                save_file_fmt=save_file_fmt,
                cfg_file_fmt=cfg_file_fmt,
            )
        case name:
            raise ValueError(
                f"Unsupported task name '{name}'. Supported tasks are 'lm' and 'resid_mlp'. {config.model_path=}, {name=}"
            )
