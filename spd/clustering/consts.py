"""Constants and shared abstractions for clustering pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

# Type definitions for merge arrays and distances
MergesAtIterArray = Int[np.ndarray, "n_ens c_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters c_components"]
DistancesMethod = Literal["perm_invariant_hamming", "jaccard"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]


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
