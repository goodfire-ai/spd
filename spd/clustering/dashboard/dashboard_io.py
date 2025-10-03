"""I/O utilities for dashboard: WandB artifacts, model setup, and result generation."""

from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from wandb.apis.public import Run

from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import SPD_CACHE_DIR


def load_wandb_artifacts(wandb_path: str) -> tuple[MergeHistory, dict[str, Any]]:
    """Download and load WandB artifacts.

    Args:
        wandb_path: WandB run path (e.g., entity/project/run_id)

    Returns:
        Tuple of (MergeHistory, run_config_dict)
    """
    api: wandb.Api = wandb.Api()
    run: Run = api.run(wandb_path)
    logger.info(f"Loaded WandB run: {run.name} ({run.id})")

    # Download merge history artifact
    logger.info("Downloading merge history artifact...")
    artifacts: list[Any] = [a for a in run.logged_artifacts() if a.type == "merge_history"]
    if not artifacts:
        raise ValueError(f"No merge_history artifacts found in run {wandb_path}")
    artifact: Any = artifacts[0]
    logger.info(f"Found artifact: {artifact.name}")

    artifact_cache_root = SPD_CACHE_DIR / "wandb_artifacts"
    artifact_cache_root.mkdir(parents=True, exist_ok=True)
    artifact_dir: str = artifact.download(root=str(artifact_cache_root))
    merge_history_path: Path = Path(artifact_dir) / "merge_history.zip"
    merge_history: MergeHistory = MergeHistory.read(merge_history_path)
    logger.info(f"Loaded merge history: {merge_history}")

    return merge_history, run.config


def setup_model_and_data(
    run_config: dict[str, Any],
    context_length: int,
    batch_size: int,
) -> tuple[ComponentModel, PreTrainedTokenizer, DataLoader[Any], Config]:
    """Set up model, tokenizer, and dataloader.

    Args:
        run_config: WandB run config dictionary
        context_length: Context length for tokenization
        batch_size: Batch size for data loading

    Returns:
        Tuple of (model, tokenizer, dataloader, spd_config)
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path: str = run_config["model_path"]
    logger.info(f"Loading model from: {model_path}")
    spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
    model: ComponentModel = ComponentModel.from_run_info(spd_run)
    model.to(device)
    model.eval()
    config: Config = spd_run.config
    tokenizer_name: str = config.tokenizer_name  # pyright: ignore[reportAssignmentType]
    logger.info(f"{tokenizer_name = }")

    # Load tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded: {tokenizer = }")

    # Create dataloader
    # TODO: read this from batches_config.json
    dataset_config: DatasetConfig = DatasetConfig(
        name="SimpleStories/SimpleStories",
        hf_tokenizer_path=tokenizer_name,
        split="train",
        n_ctx=context_length,
        is_tokenized=False,  # Text dataset
        streaming=False,
        column_name="story",
    )
    logger.info(f"Using {dataset_config = }")

    dataloader: DataLoader[Any]
    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=4,
        ddp_rank=0,
        ddp_world_size=1,
    )
    logger.info(f"Created {dataloader = }")

    return model, tokenizer, dataloader, config


def generate_model_info(
    model: ComponentModel,
    merge_history: MergeHistory,
    merge: GroupMerge,
    iteration: int,
    model_path: str,
    tokenizer_name: str,
    config_dict: dict[str, Any] | None = None,
    wandb_run_path: str | None = None,
) -> dict[str, Any]:
    """Generate model information dictionary.

    Args:
        model: The ComponentModel instance
        merge_history: MergeHistory containing component labels
        merge: GroupMerge for the current iteration
        iteration: Current iteration number
        model_path: Path to the model
        tokenizer_name: Name of the tokenizer
        config_dict: Optional config dictionary
        wandb_run_path: Optional wandb run path

    Returns:
        Dictionary containing model information
    """
    # Count unique modules from all components in the merge history
    unique_modules: set[str] = set()
    total_components: int = len(merge_history.labels)

    for label in merge_history.labels:
        module, _ = label.rsplit(":", 1)
        unique_modules.add(module)

    # Count parameters in the model
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create model info dictionary
    model_info: dict[str, Any] = {
        "total_modules": len(unique_modules),
        "total_components": total_components,
        "total_clusters": len(torch.unique(merge.group_idxs)),
        "iteration": iteration,
        "model_path": model_path,
        "tokenizer_name": tokenizer_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "component_size": getattr(model, "C", None),
        "module_list": sorted(list(unique_modules)),
    }

    # Add config information if available
    if config_dict is not None:
        model_info["config"] = config_dict

    # Add wandb run information if available
    if wandb_run_path is not None:
        model_info["wandb_run"] = wandb_run_path

    return model_info
