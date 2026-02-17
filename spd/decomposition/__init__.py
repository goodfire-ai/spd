from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from spd.autointerp.schemas import ArchitectureInfo
from spd.harvest.schemas import HarvestBatch


class Decomposition(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def architecture_info(self) -> ArchitectureInfo: ...

    @property
    @abstractmethod
    def tokenizer_name(self) -> str: ...

    @abstractmethod
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]: ...

    @abstractmethod
    def make_harvest_fn(self, device: torch.device) -> Callable[[torch.Tensor], HarvestBatch]: ...
