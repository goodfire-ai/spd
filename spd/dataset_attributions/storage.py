"""Storage classes for dataset attributions."""

import dataclasses
from collections.abc import Callable
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

    Matrix structure:
        - Rows (sources): wte tokens [0, vocab_size) + component layers [vocab_size, ...)
        - Cols (targets): component layers [0, n_components) + output tokens [n_components, ...)

    Key formats:
        - wte tokens: "wte:{token_id}" where token_id is the vocabulary index
        - component layers: "layer:c_idx" (e.g., "h.0.attn.q_proj:5")
        - output tokens: "output:{token_id}" where token_id is the vocabulary index

    Only component_layer_keys are stored explicitly. wte and output keys are
    constructed from vocab_size as f"wte:{id}" and f"output:{id}".

    attribution_matrix[i, j] = per-token average attribution FROM source i TO target j
    """

    component_layer_keys: list[str]
    """Component layer keys in order: ["h.0.attn.q_proj:0", "h.0.attn.q_proj:1", ...]"""

    vocab_size: int
    """Vocabulary size (number of wte and output tokens)"""

    attribution_matrix: Float[Tensor, "n_sources n_targets"]
    """Shape: (vocab_size + n_components, n_components + vocab_size)"""

    n_batches_processed: int
    n_tokens_processed: int
    ci_threshold: float

    _component_key_to_idx: dict[str, int] = dataclasses.field(
        default_factory=dict, repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._component_key_to_idx = {k: i for i, k in enumerate(self.component_layer_keys)}

        # Validate dimensions
        n_components = len(self.component_layer_keys)
        expected_shape = (self.vocab_size + n_components, n_components + self.vocab_size)
        assert self.attribution_matrix.shape == expected_shape, (
            f"Matrix shape {self.attribution_matrix.shape} doesn't match expected {expected_shape}"
        )

    @property
    def n_components(self) -> int:
        """Number of component layer keys."""
        return len(self.component_layer_keys)

    @property
    def n_sources(self) -> int:
        """Number of source components (wte tokens + component layers)."""
        return self.vocab_size + self.n_components

    @property
    def n_targets(self) -> int:
        """Number of target components (component layers + output tokens)."""
        return self.n_components + self.vocab_size

    def _parse_key(self, key: str) -> tuple[str, int]:
        """Parse a key into (layer, idx)."""
        layer, idx_str = key.rsplit(":", 1)
        return layer, int(idx_str)

    def _source_idx(self, key: str) -> int:
        """Get source (row) index for a key. Raises KeyError if not a valid source."""
        layer, idx = self._parse_key(key)
        if layer == "wte":
            assert 0 <= idx < self.vocab_size, (
                f"wte index {idx} out of range [0, {self.vocab_size})"
            )
            return idx
        elif layer == "output":
            raise KeyError(f"output tokens cannot be sources: {key}")
        else:
            # Component layer
            return self.vocab_size + self._component_key_to_idx[key]

    def _target_idx(self, key: str) -> int:
        """Get target (column) index for a key. Raises KeyError if not a valid target."""
        layer, idx = self._parse_key(key)
        if layer == "wte":
            raise KeyError(f"wte tokens cannot be targets: {key}")
        elif layer == "output":
            assert 0 <= idx < self.vocab_size, (
                f"output index {idx} out of range [0, {self.vocab_size})"
            )
            return self.n_components + idx
        else:
            # Component layer
            return self._component_key_to_idx[key]

    def _source_idx_to_key(self, idx: int) -> str:
        """Convert source (row) index to key."""
        if idx < self.vocab_size:
            return f"wte:{idx}"
        else:
            return self.component_layer_keys[idx - self.vocab_size]

    def _target_idx_to_key(self, idx: int) -> str:
        """Convert target (column) index to key."""
        if idx < self.n_components:
            return self.component_layer_keys[idx]
        else:
            return f"output:{idx - self.n_components}"

    def has_source(self, key: str) -> bool:
        """Check if a key can be a source (wte token or component layer)."""
        try:
            self._source_idx(key)
            return True
        except (KeyError, AssertionError):
            return False

    def has_target(self, key: str) -> bool:
        """Check if a key can be a target (component layer or output token)."""
        try:
            self._target_idx(key)
            return True
        except (KeyError, AssertionError):
            return False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_layer_keys": self.component_layer_keys,
                "vocab_size": self.vocab_size,
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
            component_layer_keys=data["component_layer_keys"],
            vocab_size=data["vocab_size"],
            attribution_matrix=data["attribution_matrix"],
            n_batches_processed=data["n_batches_processed"],
            n_tokens_processed=data["n_tokens_processed"],
            ci_threshold=data["ci_threshold"],
        )

    def get_attribution(self, source_key: str, target_key: str) -> float:
        """Get attribution strength from source to target."""
        src_idx = self._source_idx(source_key)
        tgt_idx = self._target_idx(target_key)
        return self.attribution_matrix[src_idx, tgt_idx].item()

    def _get_top_k(
        self,
        values: Tensor,
        k: int,
        sign: Literal["positive", "negative"],
        idx_to_key: Callable[[int], str],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k entries from a 1D tensor of attribution values."""
        is_positive = sign == "positive"
        top_vals, top_idxs = torch.topk(values, min(k, len(values)), largest=is_positive)

        # Filter to only values matching the requested sign
        mask = top_vals > 0 if is_positive else top_vals < 0
        top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            key = idx_to_key(idx)
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

    def get_top_sources(
        self,
        target_key: str,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k source components that attribute TO this target."""
        tgt_idx = self._target_idx(target_key)
        return self._get_top_k(
            values=self.attribution_matrix[:, tgt_idx],
            k=k,
            sign=sign,
            idx_to_key=self._source_idx_to_key,
        )

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k target components this source attributes TO."""
        src_idx = self._source_idx(source_key)
        return self._get_top_k(
            values=self.attribution_matrix[src_idx, :],
            k=k,
            sign=sign,
            idx_to_key=self._target_idx_to_key,
        )
