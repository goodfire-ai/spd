"""Types for decomposition-agnostic harvest/interp pipeline."""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torch.utils.data import Dataset

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.harvest.schemas import HarvestBatch


@dataclass
class DecompositionSpec[T]:
    """Everything needed to harvest + interpret a decomposition."""

    method: str
    decomposition_explanation: str
    component_explanations: dict[str, str]
    layers: list[tuple[str, int]]
    model: nn.Module
    tokenizer: AppTokenizer
    dataset: Dataset[T]
    harvest_fn: Callable[[T], HarvestBatch]
