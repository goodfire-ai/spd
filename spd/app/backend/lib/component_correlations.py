"""Component correlation data structures and loading.

Data classes for querying pairwise correlation metrics between components
(precision, recall, Jaccard similarity, PMI) and per-component token statistics.

Data is produced by the harvest pipeline (spd.autointerp.harvest) and loaded
from .data/harvest/<run_id>/correlations/.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.autointerp.schemas import get_correlations_dir
from spd.log import logger


@dataclass
class CorrelatedComponent:
    """A component correlated with a query component."""

    component_key: str
    score: float


@dataclass
class CorrelatedComponentWithCounts:
    """A component correlated with a query component, including raw counts for visualization."""

    component_key: str
    score: float
    count_i: int
    """Firing count of the this component"""
    count_j: int
    """Firing count of the other component"""
    count_ij: int
    """Co-occurrence count"""
    count_total: int
    """Total tokens seen"""


Metric = Literal["precision", "recall", "jaccard", "pmi"]


@dataclass
class ComponentCorrelations:
    """Stores correlation statistics between components."""

    component_keys: list[str]
    """List of component keys: ["h.0.attn.q_proj:0", "h.0.attn.q_proj:1", ...]"""
    count_i: Int[Tensor, " n_components"]
    """Firing count per component"""
    count_ij: Int[Tensor, "n_components n_components"]
    """Co-occurrence matrix: count_ij[i, j] = count of tokens where component i and j both fired"""
    count_total: int
    """Total tokens seen"""

    def _key_to_idx(self, key: str) -> int:
        return self.component_keys.index(key)

    def _compute_scores(self, component_key: str, metric: Metric) -> tuple[Tensor, int] | None:
        """Compute scores for a component. Returns (scores, component_idx)."""
        i = self._key_to_idx(component_key)

        count_this = int(self.count_i[i].item())
        if count_this == 0:
            return None

        count_others = self.count_i

        cooccurence_counts: Float[Tensor, " n_components"] = self.count_ij[i]

        match metric:
            case "precision":
                # this component's precision as a prediction of the other components
                # (or, the other components' recall as a prediction of this component)
                scores = (cooccurence_counts / count_this).nan_to_num(float("-inf"))
            case "recall":
                # this component's recall as a prediction of the other components
                # (or, the other components' precision as a prediction of this component)
                scores = (cooccurence_counts / count_others).nan_to_num(float("-inf"))
            case "jaccard":
                # jaccard = intersection / union
                intersection = cooccurence_counts
                union = count_this + count_others - cooccurence_counts
                scores = (intersection / union).nan_to_num(float("-inf"))
            case "pmi":
                # pmi = log(P(this, that) / (P(this) * P(that)))
                p_this_that = cooccurence_counts / self.count_total
                p_this = count_this / self.count_total
                p_that = count_others / self.count_total
                lift = p_this_that / (p_this * p_that)
                scores = torch.log(lift).nan_to_num(float("-inf"))

        # Mask out self and zeros with -inf
        scores[i] = float("-inf")
        scores[self.count_i == 0] = float("-inf")
        scores[cooccurence_counts == 0] = float("-inf")

        return scores, i

    def get_correlated(
        self,
        component_key: str,
        metric: Metric,
        top_k: int,
    ) -> list[CorrelatedComponent]:
        """Get top-k correlated components for a given component (vectorized).

        Args:
            component_key: The component to find correlations for (e.g., "h.0.attn.q_proj:5")
            metric: Which correlation metric to use
            top_k: Number of top correlations to return

        Returns:
            List of CorrelatedComponent sorted by score descending
        """
        result = self._compute_scores(component_key, metric)
        if result is None:
            return []

        scores, _ = result
        top_k_clamped = min(top_k, len(scores))
        top_values, top_indices = torch.topk(scores, top_k_clamped)

        # Filter out -inf (our sentinel for "excluded") but fail on unexpected non-finite values
        output = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist(), strict=True):
            if val == float("-inf"):
                continue
            assert math.isfinite(val), (
                f"Unexpected non-finite score {val} for {self.component_keys[idx]}"
            )
            output.append(CorrelatedComponent(component_key=self.component_keys[idx], score=val))

        return output

    def get_correlated_with_counts(
        self,
        component_key: str,
        metric: Metric,
        top_k: int,
        largest: bool = True,
    ) -> list[CorrelatedComponentWithCounts]:
        """Get top-k or bottom-k correlated components with raw counts for visualization.

        Args:
            component_key: The component to find correlations for (e.g., "h.0.attn.q_proj:5")
            metric: Which correlation metric to use
            top_k: Number of correlations to return
            largest: If True, return highest scores; if False, return lowest scores

        Returns:
            List of CorrelatedComponentWithCounts sorted by score (desc if largest, asc if not)
        """
        result = self._compute_scores(component_key, metric)
        if result is None:
            return []

        scores, i = result
        count_i_val = int(self.count_i[i].item())
        count_ij_row = self.count_ij[i]

        top_k_clamped = min(top_k, len(scores))
        top_values, top_indices = torch.topk(scores, top_k_clamped, largest=largest)

        output = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist(), strict=True):
            if val == float("-inf"):
                continue
            assert math.isfinite(val), (
                f"Unexpected non-finite score {val} for {self.component_keys[idx]}"
            )
            output.append(
                CorrelatedComponentWithCounts(
                    component_key=self.component_keys[idx],
                    score=val,
                    count_i=count_i_val,
                    count_j=int(self.count_i[idx].item()),
                    count_ij=int(count_ij_row[idx].item()),
                    count_total=self.count_total,
                )
            )

        return output

    def save(self, path: Path) -> None:
        """Save correlations to a .pt file."""
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
    def load(cls, path: Path) -> "ComponentCorrelations":
        """Load correlations from a .pt file using memory mapping."""
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            count_total=data["count_total"],
        )


def get_correlations_path(wandb_run_id: str) -> Path:
    """Get the path where correlations are stored for a run (from harvest output)."""
    return get_correlations_dir(wandb_run_id) / "component_correlations.pt"


def get_token_stats_path(wandb_run_id: str) -> Path:
    """Get the path where token stats are stored for a run (from harvest output)."""
    return get_correlations_dir(wandb_run_id) / "token_stats.pt"


@dataclass
class TokenPRLift:
    """Token precision, recall, lift, and PMI for a single component (input or output)."""

    top_recall: list[tuple[str, float]]  # [(token, value), ...]
    top_precision: list[tuple[str, float]]  # [(token, value), ...]
    top_lift: list[tuple[str, float]]  # [(token, lift), ...]
    top_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest positive association
    bottom_pmi: list[tuple[str, float]] | None  # [(token, pmi), ...] highest negative association


@dataclass
class ComponentTokenStats:
    """Token statistics for all components, stored as tensors.

    Tracks both input token correlations (what tokens activate this component)
    and output token correlations (what tokens does this component predict).
    """

    component_keys: list[str]  # Same ordering as ComponentCorrelations
    vocab_size: int
    n_tokens: int  # Total token positions seen

    # Input token stats: which input tokens co-occur with component firings
    # Shape: (n_components, vocab_size) - count of co-occurrences
    input_counts: Float[Tensor, "n_components vocab"]
    # Shape: (vocab_size,) - total count of each input token
    input_totals: Float[Tensor, " vocab"]

    # Output token stats: which output tokens are predicted when component fires
    # Shape: (n_components, vocab_size) - sum of probabilities
    output_counts: Float[Tensor, "n_components vocab"]
    # Shape: (vocab_size,) - total probability mass for each output token
    output_totals: Float[Tensor, " vocab"]

    # Per-component firing counts
    firing_counts: Float[Tensor, " n_components"]

    def _key_to_idx(self, key: str) -> int:
        return self.component_keys.index(key)

    @staticmethod
    def _compute_token_stats(
        counts: Float[Tensor, " vocab"],
        totals: Float[Tensor, " vocab"],
        n_tokens: int,
        firing_count: float,
        tokenizer: PreTrainedTokenizerBase,
        top_k: int,
    ) -> TokenPRLift | None:
        """Compute P/R/lift/PMI from count tensors."""
        if firing_count == 0:
            return None

        # Mask for tokens that have both co-occurred and have total counts
        valid_mask = (counts > 0) & (totals > 0)
        if not valid_mask.any():
            return None

        # Compute metrics vectorized
        recall = counts / firing_count  # P(token | firing)
        precision = torch.where(totals > 0, counts / totals, torch.zeros_like(counts))
        base_rate = firing_count / n_tokens
        lift = precision / base_rate if base_rate > 0 else torch.zeros_like(precision)

        # PMI = log(P(firing, token) / (P(firing) * P(token)))
        # P(firing, token) = counts / n_tokens
        # P(firing) = firing_count / n_tokens
        # P(token) = totals / n_tokens
        # PMI = log(counts * n_tokens / (firing_count * totals))
        pmi = torch.log(counts * n_tokens / (firing_count * totals))
        pmi = torch.where(valid_mask, pmi, torch.full_like(counts, float("-inf")))

        def get_top_k(values: Tensor, k: int, largest: bool = True) -> list[tuple[str, float]]:
            # Mask invalid values
            masked = torch.where(
                valid_mask,
                values,
                torch.full_like(values, float("-inf") if largest else float("inf")),
            )
            top_vals, top_idx = torch.topk(
                masked, min(k, int(valid_mask.sum().item())), largest=largest
            )
            result = []
            for idx, val in zip(top_idx.tolist(), top_vals.tolist(), strict=True):
                if val == float("-inf"):
                    continue
                assert math.isfinite(val), f"Unexpected non-finite score {val} for token {idx}"
                token_str = tokenizer.decode([idx])
                result.append((token_str, round(val, 3 if abs(val) < 10 else 2)))
            return result

        return TokenPRLift(
            top_recall=get_top_k(recall, top_k),
            top_precision=get_top_k(precision, top_k),
            top_lift=get_top_k(lift, top_k),
            top_pmi=get_top_k(pmi, top_k, largest=True),
            bottom_pmi=get_top_k(pmi, top_k, largest=False),
        )

    def get_input_tok_stats(
        self,
        component_key: str,
        tokenizer: PreTrainedTokenizerBase,
        top_k: int = 10,
    ) -> TokenPRLift | None:
        """Compute P/R/lift/PMI for input tokens (what tokens activate this component).

        Note: bottom_pmi is returned as empty list for input stats because it's not meaningful.
        Input stats only include tokens that co-occurred with the component (counts > 0).
        "Bottom PMI" for inputs would just show tokens that co-occurred less than expected
        among those that did co-occur - not tokens that are truly anti-correlated.
        For genuinely anti-correlated tokens (those that never co-occur), we'd need
        counts=0 which gives PMI=-inf, which isn't useful.
        """
        idx = self._key_to_idx(component_key)
        tok_stats = self._compute_token_stats(
            self.input_counts[idx],
            self.input_totals,
            self.n_tokens,
            self.firing_counts[idx].item(),
            tokenizer,
            top_k,
        )
        if tok_stats is None:
            return None
        # Clear bottom_pmi - not meaningful for input tokens (see docstring)
        return TokenPRLift(
            top_recall=tok_stats.top_recall,
            top_precision=tok_stats.top_precision,
            top_lift=tok_stats.top_lift,
            top_pmi=tok_stats.top_pmi,
            bottom_pmi=None,  # Not meaningful for inputs
        )

    def get_output_tok_stats(
        self,
        component_key: str,
        tokenizer: PreTrainedTokenizerBase,
        top_k: int = 10,
    ) -> TokenPRLift | None:
        """Compute P/R/lift/PMI for output tokens (what tokens this component predicts).

        Note: Unlike input stats, bottom_pmi IS meaningful for outputs.
        Output stats sum probability mass across all tokens at each position (softmax gives
        non-zero probability to every token). So negative PMI genuinely indicates tokens
        that are suppressed/anti-predicted when this component fires - their probability
        is lower than their baseline average.
        """
        idx = self._key_to_idx(component_key)
        return self._compute_token_stats(
            self.output_counts[idx],
            self.output_totals,
            self.n_tokens,
            self.firing_counts[idx].item(),
            tokenizer,
            top_k,
        )

    def save(self, path: Path) -> None:
        """Save token stats to a .pt file."""
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
    def load(cls, path: Path) -> "ComponentTokenStats":
        """Load token stats from a .pt file using memory mapping."""
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
