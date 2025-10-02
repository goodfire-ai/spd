"""
Loads and splits dataset into batches, returning them as an iterator.
"""

from collections.abc import Generator, Iterator
from typing import Any

import torch
from jaxtyping import Int
from muutils.spinner import SpinnerContext
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.merge_run_config import ClusteringRunConfig
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig, ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.models.component_model import ComponentModel, SPDRunInfo

BatchTensor = Int[Tensor, "batch seq"]


def split_dataset(config: ClusteringRunConfig) -> tuple[Iterator[BatchTensor], dict[str, Any]]:
    """Split a dataset into n_batches of batch_size, returning iterator and config"""
    ds: Generator[BatchTensor, None, None]
    ds_config_dict: dict[str, Any]
    match config.task_name:
        case "lm":
            ds, ds_config_dict = _get_dataloader_lm(
                model_path=config.model_path,
                batch_size=config.batch_size,
            )
        case "resid_mlp":
            ds, ds_config_dict = _get_dataloader_resid_mlp(
                model_path=config.model_path,
                batch_size=config.batch_size,
            )
        case name:
            raise ValueError(
                f"Unsupported task name '{name}'. Supported tasks are 'lm' and 'resid_mlp'. {config.model_path=}, {name=}"
            )

    # Limit iterator to n_batches
    def limited_iterator() -> Iterator[BatchTensor]:
        batch_idx: int
        batch: BatchTensor
        for batch_idx, batch in tqdm(enumerate(ds), total=config.n_batches, unit="batch"):
            if batch_idx >= config.n_batches:
                break
            yield batch

    return limited_iterator(), ds_config_dict


def _get_dataloader_lm(
    model_path: str,
    batch_size: int,
) -> tuple[Generator[BatchTensor, None, None], dict[str, Any]]:
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
                "Could not find 'pretrained_model_name' in the SPD Run config, but called `_get_dataloader_lm`"
            ) from e

        assert isinstance(cfg.task_config, LMTaskConfig), (
            f"Expected task_config to be of type LMTaskConfig since using `_get_dataloader_lm`, but got {type(cfg.task_config) = }"
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

    return (batch["input_ids"] for batch in dataloader), dataset_config.model_dump(mode="json")


def _get_dataloader_resid_mlp(
    model_path: str,
    batch_size: int,
) -> tuple[Generator[torch.Tensor, None, None], dict[str, Any]]:
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
            f"Expected task_config to be of type ResidMLPTaskConfig since using `_get_dataloader_resid_mlp`, but got {type(cfg.task_config) = }"
        )
        assert isinstance(component_model.target_model, ResidMLP), (
            f"Expected patched_model to be of type ResidMLP since using `_get_dataloader_resid_mlp`, but got {type(component_model.patched_model) = }"
        )

        assert isinstance(component_model.target_model.config, ResidMLPModelConfig), (
            f"Expected patched_model.config to be of type ResidMLPModelConfig since using `_get_dataloader_resid_mlp`, but got {type(component_model.target_model.config) = }"
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

        dataloader: DatasetGeneratedDataLoader[tuple[Tensor, Tensor]] = DatasetGeneratedDataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

    return (batch[0] for batch in dataloader), resid_mlp_dataset_kwargs
