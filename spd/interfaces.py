from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from spd.spd_types import ModelPath


@dataclass
class RunInfo[T]:
    """Base class for run information from a training run of a target model or SPD."""

    checkpoint_path: Path
    config: T

    @classmethod
    def from_path(cls, _path: ModelPath) -> "RunInfo[T]":
        """Load run info from wandb or local path.

        Args:
            _path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `tms_train_config.yaml` and `tms.pth` are in the same directory as the
                checkpoint.
        """
        raise NotImplementedError


class LoadableModel(nn.Module, ABC):
    """Base class for nn.Modules that can be loaded from a local path or wandb run id."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, _path: ModelPath) -> "LoadableModel":
        """Load a pretrained model from a local path or wandb run id."""
        raise NotImplementedError("Subclasses must implement from_pretrained method.")

    @classmethod
    @abstractmethod
    def from_run_info(cls, _run_info: RunInfo[Any]) -> "LoadableModel":
        """Load a pretrained model from a run info object."""
        raise NotImplementedError("Subclasses must implement from_run_info method.")
