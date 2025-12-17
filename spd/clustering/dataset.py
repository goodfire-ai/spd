"""Dataset loading utilities for clustering runs.

Each clustering run loads its own dataset batch, seeded by the run index.
"""

import warnings
from typing import Any

from torch.utils.data import DataLoader

from spd.clustering.consts import BatchTensor
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName


def create_dataset_loader(
    model_path: str,
    task_name: TaskName,
    batch_size: int,
    seed: int,
    **kwargs: Any,
) -> DataLoader[Any]:
    """Create a dataloader for clustering that can be iterated multiple times.

    This is more efficient than load_dataset() when you need multiple batches,
    as it creates the dataloader once and allows iteration through many batches.

    Args:
        model_path: Path to decomposed model
        task_name: Task type
        batch_size: Batch size
        seed: Random seed for dataset

    Returns:
        DataLoader that can be iterated to get multiple batches
    """
    match task_name:
        case "lm":
            return _create_lm_dataloader(
                model_path=model_path,
                batch_size=batch_size,
                seed=seed,
                **kwargs,
            )
        case "resid_mlp":
            return _create_resid_mlp_dataloader(
                model_path=model_path,
                batch_size=batch_size,
                seed=seed,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported task: {task_name}")


def load_dataset(
    model_path: str,
    task_name: TaskName,
    batch_size: int,
    seed: int,
    **kwargs: Any,
) -> BatchTensor:
    """Load a single batch for clustering.

    This is a convenience wrapper around create_dataset_loader() that extracts
    just the first batch. Use create_dataset_loader() directly if you need
    multiple batches for better efficiency.

    Args:
        model_path: Path to decomposed model
        task_name: Task type
        batch_size: Batch size
        seed: Random seed for dataset

    Returns:
        Single batch of data
    """
    dataloader = create_dataset_loader(
        model_path=model_path,
        task_name=task_name,
        batch_size=batch_size,
        seed=seed,
        **kwargs,
    )

    # Extract first batch based on task type
    batch = next(iter(dataloader))
    if task_name == "lm":
        return batch["input_ids"]
    elif task_name == "resid_mlp":
        return batch[0]  # ResidMLP returns (batch, labels) tuple
    else:
        raise ValueError(f"Unsupported task: {task_name}")


def _create_lm_dataloader(
    model_path: str,
    batch_size: int,
    seed: int,
    config_kwargs: dict[str, Any] | None = None,
) -> DataLoader[Any]:
    """Create a dataloader for language model task."""
    spd_run = SPDRunInfo.from_path(model_path)
    cfg = spd_run.config

    assert isinstance(cfg.task_config, LMTaskConfig), (
        f"Expected task_config to be of type LMTaskConfig, but got {type(cfg.task_config) = }"
    )

    try:
        pretrained_model_name: str = cfg.pretrained_model_name  # pyright: ignore[reportAssignmentType]
        assert pretrained_model_name is not None
    except Exception as e:
        raise AttributeError("Could not find 'pretrained_model_name' in the SPD Run config") from e

    config_kwargs_: dict[str, Any] = {
        **dict(
            is_tokenized=False,
            streaming=False,
        ),
        **(config_kwargs or {}),
    }

    dataset_config = DatasetConfig(
        name=cfg.task_config.dataset_name,
        hf_tokenizer_path=cfg.tokenizer_name,
        split=cfg.task_config.train_data_split,
        n_ctx=cfg.task_config.max_seq_len,
        seed=seed,  # Use run-specific seed
        column_name=cfg.task_config.column_name,
        **config_kwargs_,
    )

    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=cfg.task_config.buffer_size,
        global_seed=seed,  # Use run-specific seed
        ddp_rank=0,
        ddp_world_size=1,
    )

    return dataloader


def _create_resid_mlp_dataloader(
    model_path: str,
    batch_size: int,
    seed: int,
    config_kwargs: dict[str, Any] | None = None,
) -> DataLoader[Any]:
    """Create a dataloader for ResidMLP task."""
    from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
    from spd.utils.data_utils import DatasetGeneratedDataLoader

    spd_run = SPDRunInfo.from_path(model_path)
    cfg = spd_run.config
    component_model = ComponentModel.from_pretrained(spd_run.checkpoint_path)

    assert isinstance(cfg.task_config, ResidMLPTaskConfig), (
        f"Expected task_config to be of type ResidMLPTaskConfig, but got {type(cfg.task_config) = }"
    )
    assert isinstance(component_model.target_model, ResidMLP), (
        f"Expected target_model to be of type ResidMLP, but got {type(component_model.target_model) = }"
    )

    if config_kwargs is not None:
        if "streaming" in config_kwargs:
            warnings.warn(
                "The 'streaming' option is not supported for ResidMLPDataset and will be ignored.",
                stacklevel=1,
            )
            config_kwargs.pop("streaming")

        assert len(config_kwargs) == 0, (
            f"Unsupported config_kwargs for ResidMLPDataset: {config_kwargs=}"
        )

    # Create dataset with run-specific seed
    dataset = ResidMLPDataset(
        n_features=component_model.target_model.config.n_features,
        feature_probability=cfg.task_config.feature_probability,
        device="cpu",
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=seed,  # Use run-specific seed
        label_coeffs=None,
        data_generation_type=cfg.task_config.data_generation_type,
    )

    # Create dataloader
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
