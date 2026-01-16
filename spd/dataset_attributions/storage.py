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
    attribution_matrix[i, j] = total attribution FROM component i TO component j
    (i.e., how much source i influences target j across the dataset)
    """

    component_keys: list[str]
    """Maps flat index to 'layer:c_idx'"""
    attribution_matrix: Float[Tensor, "n_components n_components"]
    """attribution_matrix[i, j] = attribution from component i to component j"""
    n_batches_processed: int
    n_tokens_processed: int
    ci_threshold: float
    """Threshold used for filtering alive components"""
    _key_to_idx: dict[str, int] = dataclasses.field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._key_to_idx = {k: i for i, k in enumerate(self.component_keys)}

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

    def _parse_key(self, key: str) -> tuple[str, int]:
        """Parse 'layer:c_idx' into (layer, c_idx)."""
        parts = key.rsplit(":", 1)
        return parts[0], int(parts[1])

    def get_top_sources(
        self,
        target_key: str,
        k: int = 10,
        sign: Literal["positive", "negative"] = "positive",
    ) -> list[DatasetAttributionEntry]:
        """Get top-k components that attribute TO this component."""
        tgt_idx = self._key_to_idx[target_key]
        col = self.attribution_matrix[:, tgt_idx]

        if sign == "positive":
            top_vals, top_idxs = torch.topk(col, min(k, len(col)), largest=True)
            # Filter to positive values only
            mask = top_vals > 0
            top_vals, top_idxs = top_vals[mask], top_idxs[mask]
        else:
            top_vals, top_idxs = torch.topk(col, min(k, len(col)), largest=False)
            # Filter to negative values only
            mask = top_vals < 0
            top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            key = self.component_keys[idx]
            layer, c_idx = self._parse_key(key)
            results.append(
                DatasetAttributionEntry(
                    component_key=key,
                    layer=layer,
                    component_idx=c_idx,
                    value=val,
                )
            )
        return results

    def get_top_targets(
        self,
        source_key: str,
        k: int = 10,
        sign: Literal["positive", "negative"] = "positive",
    ) -> list[DatasetAttributionEntry]:
        """Get top-k components this component attributes TO."""
        src_idx = self._key_to_idx[source_key]
        row = self.attribution_matrix[src_idx, :]

        if sign == "positive":
            top_vals, top_idxs = torch.topk(row, min(k, len(row)), largest=True)
            mask = top_vals > 0
            top_vals, top_idxs = top_vals[mask], top_idxs[mask]
        else:
            top_vals, top_idxs = torch.topk(row, min(k, len(row)), largest=False)
            mask = top_vals < 0
            top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            key = self.component_keys[idx]
            layer, c_idx = self._parse_key(key)
            results.append(
                DatasetAttributionEntry(
                    component_key=key,
                    layer=layer,
                    component_idx=c_idx,
                    value=val,
                )
            )
        return results
