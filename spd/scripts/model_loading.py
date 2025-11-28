# %%
"""Shared model loading utilities for attribution scripts."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo


@dataclass
class LoadedModel:
    """Container for a loaded SPD model and its configuration."""

    model: ComponentModel
    config: Config
    run_info: SPDRunInfo
    device: str
    wandb_id: str


def load_model_from_wandb(wandb_path: str, device: str | None = None) -> LoadedModel:
    """Load a ComponentModel from a wandb run path.

    Args:
        wandb_path: Path like "wandb:goodfire/spd/runs/8ynfbr38"
        device: Device to load model on. If None, uses cuda if available.

    Returns:
        LoadedModel containing the model, config, and metadata.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_id = wandb_path.split("/")[-1]

    print(f"Using device: {device}")
    print(f"Loading model from {wandb_path}...")

    run_info = SPDRunInfo.from_path(wandb_path)
    config: Config = run_info.config
    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print(f"Number of components: {model.C}")
    print(f"Target module paths: {model.target_module_paths}")

    return LoadedModel(
        model=model,
        config=config,
        run_info=run_info,
        device=device,
        wandb_id=wandb_id,
    )


def create_data_loader_from_config(
    config: Config,
    batch_size: int,
    n_ctx: int,
    seed: int = 0,
) -> tuple[Iterable[dict[str, Any]], PreTrainedTokenizer]:
    """Create a data loader from an SPD config.

    Args:
        config: SPD Config with task configuration.
        batch_size: Batch size for the data loader.
        n_ctx: Context length.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (data_loader, tokenizer).
    """
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), "Expected LM task config"

    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=n_ctx,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=seed,
    )

    print(f"\nLoading dataset {dataset_config.name}...")
    data_loader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=seed,
    )

    return data_loader, tokenizer


def get_out_dir() -> Path:
    """Get the output directory for attribution scripts."""
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
