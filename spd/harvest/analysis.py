"""Query functions for analyzing harvest data.

These functions operate on storage classes from harvest/storage.py.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.harvest.storage import CorrelationStorage, GlobalAttributionStorage, TokenStatsStorage

Metric = Literal["precision", "recall", "jaccard", "pmi"]


@dataclass
class CorrelatedComponent:
    """A component correlated with a query component, including raw counts."""

    component_key: str
    score: float
    count_i: int
    """Firing count of the query component"""
    count_j: int
    """Firing count of this component"""
    count_ij: int
    """Co-occurrence count"""
    count_total: int
    """Total tokens seen"""


@dataclass
class TokenPRLift:
    """Token precision, recall, lift, and PMI for a single component."""

    top_recall: list[tuple[str, float]]
    top_precision: list[tuple[str, float]]
    top_lift: list[tuple[str, float]]
    top_pmi: list[tuple[str, float]]
    bottom_pmi: list[tuple[str, float]] | None


def _build_key_to_idx(component_keys: list[str]) -> dict[str, int]:
    return {k: i for i, k in enumerate(component_keys)}


def get_correlated_components(
    storage: CorrelationStorage,
    component_key: str,
    metric: Metric,
    top_k: int,
    largest: bool = True,
) -> list[CorrelatedComponent]:
    """Get top-k or bottom-k correlated components."""
    key_to_idx = _build_key_to_idx(storage.component_keys)
    i = key_to_idx[component_key]

    count_this = int(storage.count_i[i].item())
    if count_this == 0:
        return []

    count_others = storage.count_i
    cooccurrence_counts: Float[Tensor, " n_components"] = storage.count_ij[i].float()

    match metric:
        case "precision":
            scores = (cooccurrence_counts / count_this).nan_to_num(float("-inf"))
        case "recall":
            scores = (cooccurrence_counts / count_others).nan_to_num(float("-inf"))
        case "jaccard":
            intersection = cooccurrence_counts
            union = count_this + count_others - cooccurrence_counts
            scores = (intersection / union).nan_to_num(float("-inf"))
        case "pmi":
            p_this_that = cooccurrence_counts / storage.count_total
            p_this = count_this / storage.count_total
            p_that = count_others / storage.count_total
            lift = p_this_that / (p_this * p_that)
            scores = torch.log(lift).nan_to_num(float("-inf"))

    # Exclude self and inactive components
    scores[i] = float("-inf")
    scores[storage.count_i == 0] = float("-inf")
    scores[cooccurrence_counts == 0] = float("-inf")

    top_k_clamped = min(top_k, len(scores))
    top_values, top_indices = torch.topk(scores, top_k_clamped, largest=largest)

    output = []
    for idx, val in zip(top_indices.tolist(), top_values.tolist(), strict=True):
        if val == float("-inf"):
            continue
        assert math.isfinite(val), (
            f"Unexpected non-finite score {val} for {storage.component_keys[idx]}"
        )
        output.append(
            CorrelatedComponent(
                component_key=storage.component_keys[idx],
                score=val,
                count_i=count_this,
                count_j=int(storage.count_i[idx].item()),
                count_ij=int(cooccurrence_counts[idx].item()),
                count_total=storage.count_total,
            )
        )

    return output


def has_component(storage: CorrelationStorage, component_key: str) -> bool:
    """Check if a component exists in the storage."""
    key_to_idx = _build_key_to_idx(storage.component_keys)
    return component_key in key_to_idx


def get_input_token_stats(
    storage: TokenStatsStorage,
    component_key: str,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int,
) -> TokenPRLift | None:
    """Compute P/R/lift/PMI for input tokens."""
    key_to_idx = _build_key_to_idx(storage.component_keys)
    idx = key_to_idx[component_key]

    result = _compute_token_stats(
        counts=storage.input_counts[idx],
        totals=storage.input_totals,
        n_tokens=storage.n_tokens,
        firing_count=storage.firing_counts[idx].item(),
        tokenizer=tokenizer,
        top_k=top_k,
    )
    if result is None:
        return None

    # Input stats don't have bottom PMI
    return TokenPRLift(
        top_recall=result.top_recall,
        top_precision=result.top_precision,
        top_lift=result.top_lift,
        top_pmi=result.top_pmi,
        bottom_pmi=None,
    )


def get_output_token_stats(
    storage: TokenStatsStorage,
    component_key: str,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int,
) -> TokenPRLift | None:
    """Compute P/R/lift/PMI for output tokens."""
    key_to_idx = _build_key_to_idx(storage.component_keys)
    idx = key_to_idx[component_key]

    return _compute_token_stats(
        counts=storage.output_counts[idx],
        totals=storage.output_totals,
        n_tokens=storage.n_tokens,
        firing_count=storage.firing_counts[idx].item(),
        tokenizer=tokenizer,
        top_k=top_k,
    )


@dataclass
class AttributingComponent:
    """A component with attribution to/from a query component."""

    component_key: str
    attribution: float
    """Sum of (grad * activation) over the dataset"""


def get_top_attribution_sources(
    storage: GlobalAttributionStorage,
    component_key: str,
    top_k: int,
) -> tuple[list[AttributingComponent], list[AttributingComponent]]:
    """Get top components that attribute TO the query component.

    Returns:
        Tuple of (top_positive, top_negative) attribution sources.
        Positive: components that increase the query component's activation.
        Negative: components that decrease the query component's activation.
    """
    key_to_idx = _build_key_to_idx(storage.component_keys)
    if component_key not in key_to_idx:
        return [], []

    target_idx = key_to_idx[component_key]

    # Find all edges where this component is the target
    target_mask = storage.indices[1] == target_idx
    source_indices = storage.indices[0, target_mask]
    attr_values = storage.values[target_mask]

    if len(attr_values) == 0:
        return [], []

    # Get top positive sources
    positive_mask = attr_values > 0
    if positive_mask.any():
        positive_vals = attr_values[positive_mask]
        positive_srcs = source_indices[positive_mask]
        k_pos = min(top_k, len(positive_vals))
        top_pos_vals, top_pos_local_idx = torch.topk(positive_vals, k_pos, largest=True)
        top_pos_srcs = positive_srcs[top_pos_local_idx]
        top_positive = [
            AttributingComponent(
                component_key=storage.component_keys[int(src.item())],
                attribution=float(val.item()),
            )
            for src, val in zip(top_pos_srcs, top_pos_vals, strict=True)
        ]
    else:
        top_positive = []

    # Get top negative sources (most negative = largest magnitude negative)
    negative_mask = attr_values < 0
    if negative_mask.any():
        negative_vals = attr_values[negative_mask]
        negative_srcs = source_indices[negative_mask]
        k_neg = min(top_k, len(negative_vals))
        # Use largest=False to get most negative values
        top_neg_vals, top_neg_local_idx = torch.topk(negative_vals, k_neg, largest=False)
        top_neg_srcs = negative_srcs[top_neg_local_idx]
        top_negative = [
            AttributingComponent(
                component_key=storage.component_keys[int(src.item())],
                attribution=float(val.item()),
            )
            for src, val in zip(top_neg_srcs, top_neg_vals, strict=True)
        ]
    else:
        top_negative = []

    return top_positive, top_negative


def get_top_attribution_targets(
    storage: GlobalAttributionStorage,
    component_key: str,
    top_k: int,
) -> tuple[list[AttributingComponent], list[AttributingComponent]]:
    """Get top components that the query component attributes TO.

    Returns:
        Tuple of (top_positive, top_negative) attribution targets.
        Positive: components that this component increases.
        Negative: components that this component decreases.
    """
    key_to_idx = _build_key_to_idx(storage.component_keys)
    if component_key not in key_to_idx:
        return [], []

    source_idx = key_to_idx[component_key]

    # Find all edges where this component is the source
    source_mask = storage.indices[0] == source_idx
    target_indices = storage.indices[1, source_mask]
    attr_values = storage.values[source_mask]

    if len(attr_values) == 0:
        return [], []

    # Get top positive targets
    positive_mask = attr_values > 0
    if positive_mask.any():
        positive_vals = attr_values[positive_mask]
        positive_tgts = target_indices[positive_mask]
        k_pos = min(top_k, len(positive_vals))
        top_pos_vals, top_pos_local_idx = torch.topk(positive_vals, k_pos, largest=True)
        top_pos_tgts = positive_tgts[top_pos_local_idx]
        top_positive = [
            AttributingComponent(
                component_key=storage.component_keys[int(tgt.item())],
                attribution=float(val.item()),
            )
            for tgt, val in zip(top_pos_tgts, top_pos_vals, strict=True)
        ]
    else:
        top_positive = []

    # Get top negative targets
    negative_mask = attr_values < 0
    if negative_mask.any():
        negative_vals = attr_values[negative_mask]
        negative_tgts = target_indices[negative_mask]
        k_neg = min(top_k, len(negative_vals))
        top_neg_vals, top_neg_local_idx = torch.topk(negative_vals, k_neg, largest=False)
        top_neg_tgts = negative_tgts[top_neg_local_idx]
        top_negative = [
            AttributingComponent(
                component_key=storage.component_keys[int(tgt.item())],
                attribution=float(val.item()),
            )
            for tgt, val in zip(top_neg_tgts, top_neg_vals, strict=True)
        ]
    else:
        top_negative = []

    return top_positive, top_negative


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

    valid_mask = (counts > 0) & (totals > 0)
    if not valid_mask.any():
        return None

    recall = counts / firing_count
    precision = torch.where(totals > 0, counts / totals, torch.zeros_like(counts))
    base_rate = firing_count / n_tokens
    lift = precision / base_rate if base_rate > 0 else torch.zeros_like(precision)

    pmi = torch.log(counts * n_tokens / (firing_count * totals))
    pmi = torch.where(valid_mask, pmi, torch.full_like(pmi, float("-inf")))

    def get_top_k(values: Tensor, k: int, largest: bool = True) -> list[tuple[str, float]]:
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
