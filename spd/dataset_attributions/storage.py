"""Storage classes for dataset attributions."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from spd.log import logger


@dataclass
class DatasetAttributionEntry:
    """A single entry in the attribution results (component + value)."""

    component_key: str
    layer: str
    component_idx: int
    value: float


@dataclass
class DatasetAttributionStorage:
    """Dataset-aggregated attribution strengths between components.

    Stores full dense matrix for maximum flexibility.
    attribution_matrix[i, j] = per-token average attribution FROM component i TO component j
    (i.e., how much source i influences target j per token, on average across the dataset)

    Values are normalized by n_tokens_processed, making them comparable across runs
    with different numbers of batches.
    """

    component_keys: list[str]
    """Maps flat index to 'layer:c_idx'"""
    attribution_matrix: Float[Tensor, "n_components n_components"]
    """attribution_matrix[i, j] = per-token average attribution from component i to component j"""
    n_batches_processed: int
    n_tokens_processed: int
    ci_threshold: float
    """Threshold used for filtering alive components"""
    _key_to_idx: dict[str, int] = dataclasses.field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._key_to_idx = {k: i for i, k in enumerate(self.component_keys)}

    def has_component(self, component_key: str) -> bool:
        """Check if a component exists in the storage."""
        return component_key in self._key_to_idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "attribution_matrix": self.attribution_matrix.cpu(),
                "n_batches_processed": self.n_batches_processed,
                "n_tokens_processed": self.n_tokens_processed,
                "ci_threshold": self.ci_threshold,
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved dataset attributions to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "DatasetAttributionStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            attribution_matrix=data["attribution_matrix"],
            n_batches_processed=data["n_batches_processed"],
            n_tokens_processed=data["n_tokens_processed"],
            ci_threshold=data["ci_threshold"],
        )

    def get_attribution(self, source_key: str, target_key: str) -> float:
        """Get attribution strength from source to target."""
        src_idx = self._key_to_idx[source_key]
        tgt_idx = self._key_to_idx[target_key]
        return self.attribution_matrix[src_idx, tgt_idx].item()

    def _get_top_k(
        self,
        values: Tensor,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k entries from a 1D tensor of attribution values."""
        is_positive = sign == "positive"
        top_vals, top_idxs = torch.topk(values, min(k, len(values)), largest=is_positive)

        # Filter to only values matching the requested sign
        mask = top_vals > 0 if is_positive else top_vals < 0
        top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            key = self.component_keys[idx]
            layer, c_idx_str = key.rsplit(":", 1)
            results.append(
                DatasetAttributionEntry(
                    component_key=key,
                    layer=layer,
                    component_idx=int(c_idx_str),
                    value=val,
                )
            )
        return results

    def get_top_sources(
        self,
        target_key: str,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k components that attribute TO this component."""
        tgt_idx = self._key_to_idx[target_key]
        return self._get_top_k(self.attribution_matrix[:, tgt_idx], k, sign)

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k components this component attributes TO."""
        src_idx = self._key_to_idx[source_key]
        return self._get_top_k(self.attribution_matrix[src_idx, :], k, sign)
