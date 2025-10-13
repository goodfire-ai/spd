"""Constants and shared abstractions for clustering pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NewType

import numpy as np
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Merge arrays and distances (numpy-based for storage/analysis)
MergesAtIterArray = Int[np.ndarray, "n_ens n_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters n_components"]
DistancesMethod = Literal["perm_invariant_hamming", "jaccard"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]

# Component and label types (NewType for stronger type safety)
SubComponentLabel = NewType("SubComponentLabel", str)  # Format: "module_name:component_index"
SubComponentLabels = NewType("SubComponentLabels", list[str])


@dataclass(frozen=True, slots=True, kw_only=True)
class SubComponentInfo:
    """unique identifier of a subcomponent. indices can refer to dead components"""

    module: str
    index: int

    @property
    def label(self) -> SubComponentLabel:
        """Component label as 'module:index'."""
        return SubComponentLabel(f"{self.module}:{self.index}")

    @classmethod
    def from_label(cls, label: SubComponentLabel) -> "SubComponentInfo":
        """Create SubComponentInfo from a component label."""
        assert label.count(":") == 1, (
            "Invalid component label format, expected '{{module}}:{{index}}'"
        )
        module, index_str = label.rsplit(":", 1)
        return cls(module=module, index=int(index_str))


BatchId = NewType("BatchId", str)

# Path types
WandBPath = NewType("WandBPath", str)  # Format: "wandb:entity/project/run_id"

# Merge types
MergePair = NewType("MergePair", tuple[int, int])

# Tensor type aliases (torch-based for computation - TypeAlias for jaxtyping compatibility)
ActivationsTensor = Float[Tensor, "samples n_components"]
BoolActivationsTensor = Bool[Tensor, "samples n_components"]
ClusterCoactivationShaped = Float[Tensor, "k_groups k_groups"]
GroupIdxsTensor = Int[Tensor, " n_components"]
BatchTensor = Int[Tensor, "batch_size seq_len"]


class SaveableObject(ABC):
    """Abstract base class for objects that can be saved to and loaded from disk."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the object to disk at the given path."""
        ...

    @classmethod
    @abstractmethod
    def read(cls, path: Path) -> "SaveableObject":
        """Load the object from disk at the given path."""
        ...
