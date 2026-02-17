from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from spd.autointerp.schemas import ArchitectureInfo


class DecompositionAdapter(ABC):
    @property
    @abstractmethod
    def decomposition_id(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def layer_activation_sizes(self) -> list[tuple[str, int]]: ...

    @property
    @abstractmethod
    def tokenizer_name(self) -> str: ...

    @property
    @abstractmethod
    def architecture_info(self) -> ArchitectureInfo: ...

    @abstractmethod
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]: ...
