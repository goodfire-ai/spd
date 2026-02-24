"""Storage classes for dataset attributions.

Four edge types, each with its own shape:
- regular:        component → component  [tgt_c, src_c]  (signed + abs)
- embed:          embed → component      [tgt_c, vocab]  (signed + abs)
- unembed:        component → unembed    [d_model, src_c] (signed only, residual space)
- embed_unembed:  embed → unembed        [d_model, vocab] (signed only, residual space)

Output (unembed) attributions are stored in residual space. Token-level attributions
are computed on-the-fly via w_unembed projection.

Abs variants are unavailable for unembed edges because abs is a nonlinear operation
incompatible with the residual-space storage trick.
"""

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from spd.log import logger

AttrMetric = Literal["attr", "attr_abs"]


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
        regular_attr: dict[str, dict[str, Tensor]],
        regular_attr_abs: dict[str, dict[str, Tensor]],
        embed_attr: dict[str, Tensor],
        embed_attr_abs: dict[str, Tensor],
        unembed_attr: dict[str, Tensor],
        embed_unembed_attr: Tensor,
        w_unembed: Tensor,
        vocab_size: int,
        ci_threshold: float,
        n_batches_processed: int,
        n_tokens_processed: int,
    ):
        self.regular_attr = regular_attr
        self.regular_attr_abs = regular_attr_abs
        self.embed_attr = embed_attr
        self.embed_attr_abs = embed_attr_abs
        self.unembed_attr = unembed_attr
        self.embed_unembed_attr = embed_unembed_attr
        self.w_unembed = w_unembed
        self.vocab_size = vocab_size
        self.ci_threshold = ci_threshold
        self.n_batches_processed = n_batches_processed
        self.n_tokens_processed = n_tokens_processed

    @property
    def target_layers(self) -> set[str]:
        return self.regular_attr.keys() | self.embed_attr.keys()

    def _target_n_components(self, layer: str) -> int | None:
        """Number of target components for a layer, or None if not a target."""
        if layer in self.embed_attr:
            return self.embed_attr[layer].shape[0]
        if layer in self.regular_attr:
            first_source = next(iter(self.regular_attr[layer].values()))
            return first_source.shape[0]
        return None

    @property
    def n_components(self) -> int:
        total = 0
        for layer in self.target_layers:
            n = self._target_n_components(layer)
            assert n is not None
            total += n
        return total

    @staticmethod
    def _parse_key(key: str) -> tuple[str, int]:
        layer, idx_str = key.rsplit(":", 1)
        return layer, int(idx_str)

    def _select_metric(
        self, metric: AttrMetric
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        """Return (regular_dict, embed_dict) for the given metric."""
        match metric:
            case "attr":
                return self.regular_attr, self.embed_attr
            case "attr_abs":
                return self.regular_attr_abs, self.embed_attr_abs

    def get_top_sources(
        self,
        target_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
    ) -> list[DatasetAttributionEntry]:
        target_layer, target_idx = self._parse_key(target_key)

        value_segments: list[Tensor] = []
        layer_names: list[str] = []

        if target_layer == "output":
            if metric == "attr_abs":
                return []
            w = self.w_unembed[:, target_idx].to(self.embed_unembed_attr.device)

            # Component sources via unembed_attr
            for source_layer, attr_matrix in self.unembed_attr.items():
                values = w @ attr_matrix  # (d_model,) @ (d_model, src_c) → (src_c,)
                value_segments.append(values)
                layer_names.append(source_layer)

            # Embed source via embed_unembed_attr
            values = w @ self.embed_unembed_attr  # (d_model,) @ (d_model, vocab) → (vocab,)
            value_segments.append(values)
            layer_names.append("embed")
        else:
            regular, embed = self._select_metric(metric)

            # Component sources
            if target_layer in regular:
                for source_layer, attr_matrix in regular[target_layer].items():
                    values = attr_matrix[target_idx, :]
                    value_segments.append(values)
                    layer_names.append(source_layer)

            # Embed source
            if target_layer in embed:
                values = embed[target_layer][target_idx, :]
                value_segments.append(values)
                layer_names.append("embed")

        return self._top_k_from_segments(value_segments, layer_names, k, sign)

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
        include_outputs: bool = True,
    ) -> list[DatasetAttributionEntry]:
        source_layer, source_idx = self._parse_key(source_key)

        value_segments: list[Tensor] = []
        layer_names: list[str] = []

        if source_layer == "embed":
            regular, embed = self._select_metric(metric)

            for target_layer, attr_matrix in embed.items():
                values = attr_matrix[:, source_idx]  # (tgt_c,)
                value_segments.append(values)
                layer_names.append(target_layer)

            if include_outputs and metric == "attr":
                residual = self.embed_unembed_attr[:, source_idx]  # (d_model,)
                values = residual @ self.w_unembed  # (d_model,) @ (d_model, vocab) → (vocab,)
                value_segments.append(values)
                layer_names.append("output")
        else:
            regular, embed = self._select_metric(metric)

            for target_layer, sources in regular.items():
                if source_layer not in sources:
                    continue
                values = sources[source_layer][:, source_idx]  # (tgt_c,)
                value_segments.append(values)
                layer_names.append(target_layer)

            if include_outputs and metric == "attr" and source_layer in self.unembed_attr:
                residual = self.unembed_attr[source_layer][:, source_idx]  # (d_model,)
                values = residual @ self.w_unembed  # (d_model,) @ (d_model, vocab) → (vocab,)
                value_segments.append(values)
                layer_names.append("output")

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

        to_cpu_nested = partial(_d_map_nested, lambda x: x.cpu())
        to_cpu_flat = partial(_d_map, lambda x: x.cpu())

        torch.save(
            {
                "regular_attr": to_cpu_nested(self.regular_attr),
                "regular_attr_abs": to_cpu_nested(self.regular_attr_abs),
                "embed_attr": to_cpu_flat(self.embed_attr),
                "embed_attr_abs": to_cpu_flat(self.embed_attr_abs),
                "unembed_attr": to_cpu_flat(self.unembed_attr),
                "embed_unembed_attr": self.embed_unembed_attr.cpu(),
                "w_unembed": self.w_unembed.cpu(),
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
            regular_attr=data["regular_attr"],
            regular_attr_abs=data["regular_attr_abs"],
            embed_attr=data["embed_attr"],
            embed_attr_abs=data["embed_attr_abs"],
            unembed_attr=data["unembed_attr"],
            embed_unembed_attr=data["embed_unembed_attr"],
            w_unembed=data["w_unembed"],
            vocab_size=data["vocab_size"],
            ci_threshold=data["ci_threshold"],
            n_batches_processed=data["n_batches_processed"],
            n_tokens_processed=data["n_tokens_processed"],
        )

    @classmethod
    def merge(cls, paths: list[Path]) -> "DatasetAttributionStorage":
        """Merge partial attribution files from parallel workers.

        Values are treated as means — merge is weighted average by n_tokens.
        """
        assert paths, "No files to merge"

        first = cls.load(paths[0])
        n = first.n_tokens_processed

        denorm_nested = partial(_d_map_nested, lambda x: (x * n).double())
        denorm_flat = partial(_d_map, lambda x: (x * n).double())

        total_regular = denorm_nested(first.regular_attr)
        total_regular_abs = denorm_nested(first.regular_attr_abs)
        total_embed = denorm_flat(first.embed_attr)
        total_embed_abs = denorm_flat(first.embed_attr_abs)
        total_unembed = denorm_flat(first.unembed_attr)
        total_embed_unembed = (first.embed_unembed_attr * n).double()
        total_tokens = n
        total_batches = first.n_batches_processed

        for path in paths[1:]:
            storage = cls.load(path)
            assert storage.ci_threshold == first.ci_threshold, "CI threshold mismatch"
            n = storage.n_tokens_processed

            for target, sources in storage.regular_attr.items():
                for source, tensor in sources.items():
                    total_regular[target][source] += (tensor * n).double()
                    total_regular_abs[target][source] += (
                        storage.regular_attr_abs[target][source] * n
                    ).double()

            for target, tensor in storage.embed_attr.items():
                total_embed[target] += (tensor * n).double()
                total_embed_abs[target] += (storage.embed_attr_abs[target] * n).double()

            for source, tensor in storage.unembed_attr.items():
                total_unembed[source] += (tensor * n).double()

            total_embed_unembed += (storage.embed_unembed_attr * n).double()
            total_tokens += n
            total_batches += storage.n_batches_processed

        norm_nested = partial(_d_map_nested, lambda x: (x / total_tokens).float())
        norm_flat = partial(_d_map, lambda x: (x / total_tokens).float())

        return cls(
            regular_attr=norm_nested(total_regular),
            regular_attr_abs=norm_nested(total_regular_abs),
            embed_attr=norm_flat(total_embed),
            embed_attr_abs=norm_flat(total_embed_abs),
            unembed_attr=norm_flat(total_unembed),
            embed_unembed_attr=(total_embed_unembed / total_tokens).float(),
            w_unembed=first.w_unembed,
            vocab_size=first.vocab_size,
            ci_threshold=first.ci_threshold,
            n_batches_processed=total_batches,
            n_tokens_processed=total_tokens,
        )


def _d_map_nested(
    f: Callable[[Tensor], Tensor], d: dict[str, dict[str, Tensor]]
) -> dict[str, dict[str, Tensor]]:
    return {
        target: {source: f(v) for source, v in sources.items()} for target, sources in d.items()
    }


def _d_map(f: Callable[[Tensor], Tensor], d: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: f(v) for k, v in d.items()}
