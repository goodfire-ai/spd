"""Raw storage classes for harvest data.

These are simple data containers with save/load methods.
For query functionality, see harvest/analysis.py.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.log import logger


@dataclass
class CorrelationStorage:
    """Raw correlation data between components."""

    component_keys: list[str]
    count_i: Int[Tensor, " n_components"]
    """Firing count per component"""
    count_ij: Int[Tensor, "n_components n_components"]
    """Co-occurrence matrix: count_ij[i, j] = count of tokens where both fired"""
    count_total: int
    """Total tokens seen"""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "count_i": self.count_i.cpu(),
                "count_ij": self.count_ij.cpu(),
                "count_total": self.count_total,
            },
            path,
        )
        logger.info(f"Saved component correlations to {path}")

    @classmethod
    def load(cls, path: Path) -> "CorrelationStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            count_total=data["count_total"],
        )


@dataclass
class TokenStatsStorage:
    """Raw token statistics for all components.

    Input stats are hard counts (token appeared when component fired).
    Output stats are probability mass (sum of probs assigned to token when component fired).
    Both are used identically in analysis (precision/recall/PMI computations).
    """

    component_keys: list[str]
    vocab_size: int
    n_tokens: int

    input_counts: Float[Tensor, "n_components vocab"]
    input_totals: Float[Tensor, " vocab"]
    output_counts: Float[Tensor, "n_components vocab"]
    """Probability mass, not hard counts - but used the same way in analysis."""
    output_totals: Float[Tensor, " vocab"]
    firing_counts: Float[Tensor, " n_components"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "vocab_size": self.vocab_size,
                "n_tokens": self.n_tokens,
                "input_counts": self.input_counts.cpu(),
                "input_totals": self.input_totals.cpu(),
                "output_counts": self.output_counts.cpu(),
                "output_totals": self.output_totals.cpu(),
                "firing_counts": self.firing_counts.cpu(),
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved token stats to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "TokenStatsStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            vocab_size=data["vocab_size"],
            n_tokens=data["n_tokens"],
            input_counts=data["input_counts"],
            input_totals=data["input_totals"],
            output_counts=data["output_counts"],
            output_totals=data["output_totals"],
            firing_counts=data["firing_counts"],
        )


@dataclass
class GlobalAttributionStorage:
    """Global (dataset-summed) attribution data between components.

    Stores the sum of attribution values (grad * activation) between component pairs
    over the entire dataset. Uses sparse COO format since most pairs have zero attribution.

    Attribution from source i to target j: attribution_matrix[i, j] = sum over dataset of
    (d_target_j / d_source_i) * activation_source_i
    """

    component_keys: list[str]
    """Ordered list of component keys (e.g., 'h.0.mlp.c_fc:5')"""
    indices: Int[Tensor, "2 nnz"]
    """COO format: indices[0] = source indices, indices[1] = target indices"""
    values: Float[Tensor, " nnz"]
    """Attribution values for each (source, target) pair"""
    n_components: int
    """Total number of components (matrix dimension)"""
    n_samples: int
    """Number of samples (tokens * seq_len) processed"""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "indices": self.indices.cpu(),
                "values": self.values.cpu(),
                "n_components": self.n_components,
                "n_samples": self.n_samples,
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        nnz = self.values.numel()
        logger.info(
            f"Saved global attributions to {path} ({size_mb:.1f} MB, {nnz:,} non-zero entries)"
        )

    @classmethod
    def load(cls, path: Path) -> "GlobalAttributionStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            indices=data["indices"],
            values=data["values"],
            n_components=data["n_components"],
            n_samples=data["n_samples"],
        )

    def to_dense(self) -> Float[Tensor, "n_components n_components"]:
        """Convert to dense matrix for analysis."""
        dense = torch.zeros(self.n_components, self.n_components)
        dense[self.indices[0], self.indices[1]] = self.values
        return dense
