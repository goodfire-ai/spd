from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

# Type alias for readability
WandbStr = str
class Config:
    pass


@dataclass
class SPDRunInfo:
    checkpoint_path: Path
    config: Config

    @classmethod
    def from_path(cls, path: Path | WandbStr) -> "SPDRunInfo":
        """
        Load from local path or wandb str ("wandb:goodfire/spd/ewfe2h4")
        """
        pass


class ComponentModel(nn.Module):
    @classmethod
    def from_run_info(cls, run_info: SPDRunInfo) -> "ComponentModel":
        """
        Runs SPDRunInfo.from_path() then ComponentModel.from_run_info()
        """
        pass

    @classmethod
    def from_pretrained(cls, path: Path | WandbStr) -> "ComponentModel":
        """
        Runs SPDRunInfo.from_path() then ComponentModel.from_run_info()
        """
        pass


