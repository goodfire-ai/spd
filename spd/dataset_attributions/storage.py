"""Storage classes for dataset attributions.

Stored as nested dicts: attrs[target_layer][source_layer] = Tensor[target_d, source_d]

Three attribution metrics are stored:
- attr: mean attribution of source to target (signed)
- attr_abs: mean attribution of source to |target| (always positive for positive activations)
- mean_squared_attr: mean of squared attributions (pre-sqrt, for mergeable RMS)

For output targets, target_d = d_model (residual stream dimension).
Output token attributions are computed on-the-fly via w_unembed.
"""

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from spd.log import logger

AttrDict = dict[str, dict[str, Tensor]]
AttrMetric = Literal["attr", "attr_abs", "mean_squared_attr"]


@dataclass
class DatasetAttributionEntry:
    """A single entry in the attribution results (component + value)."""

    component_key: str
    layer: str
    component_idx: int
    value: float


class DatasetAttributionStorage:
    """Dataset-aggregated attribution strengths between components.

    All layer names use canonical addressing (e.g., "embed", "0.glu.up", "output").

    Key formats:
        - embed tokens: "embed:{token_id}"
        - component layers: "canonical_layer:c_idx" (e.g., "0.glu.up:5")
        - output tokens: "output:{token_id}"
    """

    def __init__(
        self,
        attr: AttrDict,
        attr_abs: AttrDict,
        mean_squared_attr: AttrDict,
        vocab_size: int,
        ci_threshold: float,
        n_batches_processed: int,
        n_tokens_processed: int,
    ):
        self.attr = attr
        self.attr_abs = attr_abs
        self.mean_squared_attr = mean_squared_attr
        self.vocab_size = vocab_size
        self.ci_threshold = ci_threshold
        self.n_batches_processed = n_batches_processed
        self.n_tokens_processed = n_tokens_processed

    @property
    def n_components(self) -> int:
        total = 0
        for target_layer in self.attr:
            if target_layer == "output":
                continue
            first_source = next(iter(self.attr[target_layer].values()))
            total += first_source.shape[0]
        return total

    @staticmethod
    def _parse_key(key: str) -> tuple[str, int]:
        layer, idx_str = key.rsplit(":", 1)
        return layer, int(idx_str)

    def has_source(self, key: str) -> bool:
        layer, idx = self._parse_key(key)
        if layer == "output":
            return False
        for target_sources in self.attr.values():
            if layer in target_sources:
                return 0 <= idx < target_sources[layer].shape[1]
        return False

    def has_target(self, key: str) -> bool:
        layer, idx = self._parse_key(key)
        match layer:
            case "embed":
                return False
            case "output":
                return 0 <= idx < self.vocab_size
            case _:
                if layer not in self.attr:
                    return False
                first_source = next(iter(self.attr[layer].values()))
                return 0 <= idx < first_source.shape[0]

    def _get_attr_dict(self, metric: AttrMetric) -> AttrDict:
        match metric:
            case "attr":
                return self.attr
            case "attr_abs":
                return self.attr_abs
            case "mean_squared_attr":
                return self.mean_squared_attr

    def get_attribution(
        self,
        source_key: str,
        target_key: str,
        metric: AttrMetric,
        w_unembed: Tensor | None = None,
    ) -> float:
        source_layer, source_idx = self._parse_key(source_key)
        target_layer, target_idx = self._parse_key(target_key)
        assert source_layer != "output", f"output tokens cannot be sources: {source_key}"

        attrs = self._get_attr_dict(metric)
        attr_matrix = attrs[target_layer][source_layer]

        if target_layer == "output":
            assert w_unembed is not None, "w_unembed required for output target queries"
            w_unembed = w_unembed.to(attr_matrix.device)
            return (attr_matrix[:, source_idx] @ w_unembed[:, target_idx]).item()

        return attr_matrix[target_idx, source_idx].item()

    def get_top_sources(
        self,
        target_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
        w_unembed: Tensor | None = None,
    ) -> list[DatasetAttributionEntry]:
        target_layer, target_idx = self._parse_key(target_key)
        attrs = self._get_attr_dict(metric)

        if target_layer == "output":
            assert w_unembed is not None, "w_unembed required for output target queries"

        value_segments: list[Tensor] = []
        layer_names: list[str] = []

        for source_layer, attr_matrix in attrs[target_layer].items():
            if target_layer == "output":
                assert w_unembed is not None
                w = w_unembed.to(attr_matrix.device)
                values = w[:, target_idx] @ attr_matrix
            else:
                values = attr_matrix[target_idx, :]

            value_segments.append(values)
            layer_names.append(source_layer)

        return self._top_k_from_segments(value_segments, layer_names, k, sign)

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
        w_unembed: Tensor | None = None,
        include_outputs: bool = True,
    ) -> list[DatasetAttributionEntry]:
        source_layer, source_idx = self._parse_key(source_key)
        attrs = self._get_attr_dict(metric)

        value_segments: list[Tensor] = []
        layer_names: list[str] = []

        for target_layer, sources in attrs.items():
            if source_layer not in sources:
                continue

            attr_matrix = sources[source_layer]

            if target_layer == "output":
                if not include_outputs:
                    continue
                assert w_unembed is not None, "w_unembed required when include_outputs=True"
                w = w_unembed.to(attr_matrix.device)
                values = attr_matrix[:, source_idx] @ w
            else:
                values = attr_matrix[:, source_idx]

            value_segments.append(values)
            layer_names.append(target_layer)

        return self._top_k_from_segments(value_segments, layer_names, k, sign)

    def _top_k_from_segments(
        self,
        value_segments: list[Tensor],
        layer_names: list[str],
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        if not value_segments:
            return []

        all_values = torch.cat(value_segments)
        offsets = [0]
        for seg in value_segments:
            offsets.append(offsets[-1] + len(seg))

        is_positive = sign == "positive"
        top_vals, top_idxs = torch.topk(all_values, min(k, len(all_values)), largest=is_positive)

        mask = top_vals > 0 if is_positive else top_vals < 0
        top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for flat_idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            seg_idx = bisect.bisect_right(offsets, flat_idx) - 1
            local_idx = flat_idx - offsets[seg_idx]
            layer = layer_names[seg_idx]
            results.append(
                DatasetAttributionEntry(
                    component_key=f"{layer}:{local_idx}",
                    layer=layer,
                    component_idx=local_idx,
                    value=val,
                )
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def to_cpu(d: AttrDict) -> AttrDict:
            return {
                target: {source: tensor.cpu() for source, tensor in sources.items()}
                for target, sources in d.items()
            }

        torch.save(
            {
                "attr": to_cpu(self.attr),
                "attr_abs": to_cpu(self.attr_abs),
                "mean_squared_attr": to_cpu(self.mean_squared_attr),
                "vocab_size": self.vocab_size,
                "ci_threshold": self.ci_threshold,
                "n_batches_processed": self.n_batches_processed,
                "n_tokens_processed": self.n_tokens_processed,
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved dataset attributions to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "DatasetAttributionStorage":
        data = torch.load(path, weights_only=True)
        return cls(
            attr=data["attr"],
            attr_abs=data["attr_abs"],
            mean_squared_attr=data["mean_squared_attr"],
            vocab_size=data["vocab_size"],
            ci_threshold=data["ci_threshold"],
            n_batches_processed=data["n_batches_processed"],
            n_tokens_processed=data["n_tokens_processed"],
        )

    @classmethod
    def merge(cls, paths: list[Path]) -> "DatasetAttributionStorage":
        """Merge partial attribution files from parallel workers.

        All three metrics are means, so merge is weighted average by n_tokens.
        (mean_squared_attr is E[x²], not sqrt(E[x²]), so this works.)
        """
        assert paths, "No files to merge"

        first = cls.load(paths[0])
        n = first.n_tokens_processed

        def denormalize(d: AttrDict, n_tokens: int) -> AttrDict:
            return {
                target: {source: (tensor * n_tokens).double() for source, tensor in sources.items()}
                for target, sources in d.items()
            }

        total_attr = denormalize(first.attr, n)
        total_attr_abs = denormalize(first.attr_abs, n)
        total_mean_squared_attr = denormalize(first.mean_squared_attr, n)
        total_tokens = n
        total_batches = first.n_batches_processed

        for path in paths[1:]:
            storage = cls.load(path)
            assert storage.ci_threshold == first.ci_threshold, "CI threshold mismatch"
            assert storage.attr.keys() == first.attr.keys(), "Target layer mismatch"
            n = storage.n_tokens_processed

            for target, sources in storage.attr.items():
                for source, tensor in sources.items():
                    total_attr[target][source] += (tensor * n).double()
                    total_attr_abs[target][source] += (
                        storage.attr_abs[target][source] * n
                    ).double()
                    total_mean_squared_attr[target][source] += (
                        storage.mean_squared_attr[target][source] * n
                    ).double()
            total_tokens += n
            total_batches += storage.n_batches_processed

        def normalize(d: AttrDict) -> AttrDict:
            return {
                target: {
                    source: (tensor / total_tokens).float() for source, tensor in sources.items()
                }
                for target, sources in d.items()
            }

        return cls(
            attr=normalize(total_attr),
            attr_abs=normalize(total_attr_abs),
            mean_squared_attr=normalize(total_mean_squared_attr),
            vocab_size=first.vocab_size,
            ci_threshold=first.ci_threshold,
            n_batches_processed=total_batches,
            n_tokens_processed=total_tokens,
        )
