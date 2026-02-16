"""Types for decomposition-agnostic harvest/interp pipeline."""

from dataclasses import dataclass
from typing import Any, Protocol

from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import Dataset

from spd.app.backend.app_tokenizer import AppTokenizer

BatchLike = dict[str, Any] | tuple[Tensor, ...] | Int[Tensor, "batch seq"]


class ActivationFn(Protocol):
    """Maps a batch to per-token activation values for each component."""

    def __call__(
        self,
        model: nn.Module,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]: ...


class OutputProbsFn(Protocol):
    """Computes next-token probabilities for each position in a batch."""

    def __call__(
        self,
        model: nn.Module,
        batch: BatchLike,
    ) -> Float[Tensor, "batch seq vocab"]: ...


class BatchToTokensFn(Protocol):
    """Extracts integer token IDs [batch, seq] from a dataset batch payload."""

    def __call__(self, batch: BatchLike) -> Int[Tensor, "batch seq"]: ...


class ComponentActsFn(Protocol):
    """Optional per-token component strength for context display."""

    def __call__(
        self,
        model: nn.Module,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]: ...


@dataclass
class DecompositionSpec:
    """Everything needed to harvest + interpret a decomposition."""

    method: str
    decomposition_explanation: str
    component_explanations: dict[str, str]
    model: nn.Module
    tokenizer: AppTokenizer
    dataset: Dataset[Any]
    activation_fn: ActivationFn
    binarise_threshold: float
    output_probs_fn: OutputProbsFn | None = None
    batch_to_tokens_fn: BatchToTokensFn | None = None
    component_acts_fn: ComponentActsFn | None = None
    component_order: list[str] | None = None
