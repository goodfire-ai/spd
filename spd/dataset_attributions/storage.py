"""Storage classes for dataset attributions.

Uses a residual-based storage approach for scalability:
- Component targets: stored directly in source_to_component matrix
- Output targets: stored as attributions to residual stream, computed on-the-fly via w_unembed
"""

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

    Uses residual-based storage for scalability with large vocabularies:
    - source_to_component: direct attributions to component targets
    - source_to_out_residual: attributions to output residual stream (for computing output attributions)

    Output attributions are computed on-the-fly: attr[src, output_token] = out_residual[src] @ w_unembed[:, token]

    Source indexing (rows):
        - [0, vocab_size): wte tokens
        - [vocab_size, vocab_size + n_components): component layers

    Target indexing:
        - Component targets: [0, n_components) in source_to_component
        - Output targets: computed via source_to_out_residual @ w_unembed

    Key formats:
        - wte tokens: "wte:{token_id}"
        - component layers: "layer:c_idx" (e.g., "h.0.attn.q_proj:5")
        - output tokens: "output:{token_id}"
    """

    component_layer_keys: list[str]
    """Component layer keys in order: ["h.0.attn.q_proj:0", "h.0.attn.q_proj:1", ...]"""

    vocab_size: int
    """Vocabulary size (number of wte and output tokens)"""

    d_model: int
    """Model hidden dimension (residual stream size)"""

    source_to_component: Float[Tensor, "n_sources n_components"]
    """Attributions from sources to component targets. Shape: (vocab_size + n_components, n_components)"""

    source_to_out_residual: Float[Tensor, "n_sources d_model"]
    """Attributions from sources to output residual dimensions. Shape: (vocab_size + n_components, d_model)"""

    n_batches_processed: int
    n_tokens_processed: int
    ci_threshold: float

    _component_key_to_idx: dict[str, int] = dataclasses.field(
        default_factory=dict, repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._component_key_to_idx = {k: i for i, k in enumerate(self.component_layer_keys)}

        n_components = len(self.component_layer_keys)
        n_sources = self.vocab_size + n_components

        expected_comp_shape = (n_sources, n_components)
        assert self.source_to_component.shape == expected_comp_shape, (
            f"source_to_component shape {self.source_to_component.shape} "
            f"doesn't match expected {expected_comp_shape}"
        )

        expected_resid_shape = (n_sources, self.d_model)
        assert self.source_to_out_residual.shape == expected_resid_shape, (
            f"source_to_out_residual shape {self.source_to_out_residual.shape} "
            f"doesn't match expected {expected_resid_shape}"
        )

    @property
    def n_components(self) -> int:
        return len(self.component_layer_keys)

    @property
    def n_sources(self) -> int:
        return self.vocab_size + self.n_components

    def _parse_key(self, key: str) -> tuple[str, int]:
        """Parse a key into (layer, idx)."""
        layer, idx_str = key.rsplit(":", 1)
        return layer, int(idx_str)

    def _source_idx(self, key: str) -> int:
        """Get source (row) index for a key. Raises KeyError if not a valid source."""
        layer, idx = self._parse_key(key)
        match layer:
            case "wte":
                assert 0 <= idx < self.vocab_size, (
                    f"wte index {idx} out of range [0, {self.vocab_size})"
                )
                return idx
            case "output":
                raise KeyError(f"output tokens cannot be sources: {key}")
            case _:
                return self.vocab_size + self._component_key_to_idx[key]

    def _component_target_idx(self, key: str) -> int:
        """Get target index for a component key. Raises KeyError if output or invalid."""
        if key.startswith(("wte:", "output:")):
            raise KeyError(f"Not a component target: {key}")
        return self._component_key_to_idx[key]

    def _source_idx_to_key(self, idx: int) -> str:
        """Convert source (row) index to key."""
        if idx < self.vocab_size:
            return f"wte:{idx}"
        return self.component_layer_keys[idx - self.vocab_size]

    def _component_target_idx_to_key(self, idx: int) -> str:
        """Convert component target index to key."""
        return self.component_layer_keys[idx]

    def _output_target_idx_to_key(self, idx: int) -> str:
        """Convert output token index to key."""
        return f"output:{idx}"

    def _is_output_target(self, key: str) -> bool:
        """Check if key is an output target."""
        return key.startswith("output:")

    def _output_token_id(self, key: str) -> int:
        """Extract token_id from an output key like 'output:123'. Asserts valid range."""
        _, token_id = self._parse_key(key)
        assert 0 <= token_id < self.vocab_size, f"output index {token_id} out of range"
        return token_id

    def has_source(self, key: str) -> bool:
        """Check if a key can be a source (wte token or component layer)."""
        layer, idx = self._parse_key(key)
        match layer:
            case "wte":
                return 0 <= idx < self.vocab_size
            case "output":
                return False
            case _:
                return key in self._component_key_to_idx

    def has_target(self, key: str) -> bool:
        """Check if a key can be a target (component layer or output token)."""
        layer, idx = self._parse_key(key)
        match layer:
            case "wte":
                return False
            case "output":
                return 0 <= idx < self.vocab_size
            case _:
                return key in self._component_key_to_idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_layer_keys": self.component_layer_keys,
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "source_to_component": self.source_to_component.cpu(),
                "source_to_out_residual": self.source_to_out_residual.cpu(),
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
            d_model=data["d_model"],
            source_to_component=data["source_to_component"],
            source_to_out_residual=data["source_to_out_residual"],
            n_batches_processed=data["n_batches_processed"],
            n_tokens_processed=data["n_tokens_processed"],
            ci_threshold=data["ci_threshold"],
        )

    def get_attribution(
        self,
        source_key: str,
        target_key: str,
        w_unembed: Float[Tensor, "d_model vocab"] | None,
    ) -> float:
        """Get attribution strength from source to target.

        Args:
            source_key: Source component key (wte or component layer)
            target_key: Target component key (component layer or output token)
            w_unembed: Unembedding matrix, required if target is an output token
        """
        src_idx = self._source_idx(source_key)

        if self._is_output_target(target_key):
            assert w_unembed is not None, "w_unembed required for output target queries"
            token_id = self._output_token_id(target_key)
            w_unembed = w_unembed.to(self.source_to_out_residual.device)
            return (self.source_to_out_residual[src_idx] @ w_unembed[:, token_id]).item()

        tgt_idx = self._component_target_idx(target_key)
        return self.source_to_component[src_idx, tgt_idx].item()

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
        w_unembed: Float[Tensor, "d_model vocab"] | None = None,
    ) -> list[DatasetAttributionEntry]:
        """Get top-k source components that attribute TO this target.

        Args:
            target_key: Target component key (component layer or output token)
            k: Number of top sources to return
            sign: "positive" for strongest positive, "negative" for strongest negative
            w_unembed: Unembedding matrix, required if target is an output token
        """
        if self._is_output_target(target_key):
            assert w_unembed is not None, "w_unembed required for output target queries"
            token_id = self._output_token_id(target_key)
            w_unembed = w_unembed.to(self.source_to_out_residual.device)
            values = self.source_to_out_residual @ w_unembed[:, token_id]  # (n_sources,)
        else:
            tgt_idx = self._component_target_idx(target_key)
            values = self.source_to_component[:, tgt_idx]

        return self._get_top_k(values, k, sign, self._source_idx_to_key)

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        w_unembed: Float[Tensor, "d_model vocab"] | None,
        include_outputs: bool = True,
    ) -> list[DatasetAttributionEntry]:
        """Get top-k target components this source attributes TO.

        Args:
            source_key: Source component key (wte or component layer)
            k: Number of top targets to return
            sign: "positive" for strongest positive, "negative" for strongest negative
            w_unembed: Unembedding matrix, required if include_outputs=True
            include_outputs: Whether to include output tokens in results
        """
        src_idx = self._source_idx(source_key)
        comp_values = self.source_to_component[src_idx, :]  # (n_components,)

        if include_outputs:
            assert w_unembed is not None, "w_unembed required when include_outputs=True"
            # Compute attributions to all output tokens
            w_unembed = w_unembed.to(self.source_to_out_residual.device)
            output_values = self.source_to_out_residual[src_idx, :] @ w_unembed  # (vocab,)
            all_values = torch.cat([comp_values, output_values])

            def combined_idx_to_key(idx: int) -> str:
                if idx < self.n_components:
                    return self._component_target_idx_to_key(idx)
                return self._output_target_idx_to_key(idx - self.n_components)

            return self._get_top_k(all_values, k, sign, combined_idx_to_key)

        return self._get_top_k(comp_values, k, sign, self._component_target_idx_to_key)

    def get_top_component_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k component targets (excluding outputs) this source attributes TO.

        Convenience method that doesn't require w_unembed.
        """
        return self.get_top_targets(source_key, k, sign, w_unembed=None, include_outputs=False)

    def get_top_output_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        w_unembed: Float[Tensor, "d_model vocab"],
    ) -> list[DatasetAttributionEntry]:
        """Get top-k output token targets this source attributes TO."""
        src_idx = self._source_idx(source_key)
        w_unembed = w_unembed.to(self.source_to_out_residual.device)
        output_values = self.source_to_out_residual[src_idx, :] @ w_unembed  # (vocab,)
        return self._get_top_k(output_values, k, sign, self._output_target_idx_to_key)
