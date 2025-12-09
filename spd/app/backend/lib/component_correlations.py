"""Component correlation computation and storage.

Computes pairwise correlation metrics between components based on their
co-occurrence patterns across a dataset. Metrics include precision, recall,
F1, Jaccard similarity, and PMI.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data

CORRELATIONS_DIR = Path(__file__).parent.parent.parent.parent.parent / "correlations"

Metric = Literal["precision", "recall", "f1", "jaccard", "pmi"]

# Type for metric functions: (count_i, count_j, count_ij, n_tokens) -> score
MetricFn = Callable[[float, float, float, int], float]


def _precision(count_i: float, count_j: float, count_ij: float, n_tokens: int) -> float:  # pyright: ignore[reportUnusedParameter]
    """P(j fires | i fires) = count_ij / count_i"""
    return count_ij / count_i


def _recall(count_i: float, count_j: float, count_ij: float, n_tokens: int) -> float:  # pyright: ignore[reportUnusedParameter]
    """P(i fires | j fires) = count_ij / count_j"""
    return count_ij / count_j


def _f1(count_i: float, count_j: float, count_ij: float, n_tokens: int) -> float:  # pyright: ignore[reportUnusedParameter]
    """Harmonic mean of precision and recall."""
    precision = count_ij / count_i
    recall = count_ij / count_j
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard(count_i: float, count_j: float, count_ij: float, n_tokens: int) -> float:  # pyright: ignore[reportUnusedParameter]
    """|i ∩ j| / |i ∪ j|"""
    union = count_i + count_j - count_ij
    if union == 0:
        return 0.0
    return count_ij / union


def _pmi(count_i: float, count_j: float, count_ij: float, n_tokens: int) -> float:
    """Pointwise mutual information: log(P(i,j) / (P(i) * P(j)))"""
    if count_ij == 0:
        return float("-inf")
    return math.log(count_ij * n_tokens / (count_i * count_j))


METRIC_FNS: dict[Metric, MetricFn] = {
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "jaccard": _jaccard,
    "pmi": _pmi,
}


@dataclass
class CorrelatedComponent:
    """A component correlated with a query component."""

    component_key: str
    score: float


@dataclass
class ComponentCorrelations:
    """Stores correlation statistics between components.

    All counts are stored as upper-triangular to avoid redundancy.
    The diagonal of count_ij represents self-co-occurrence (same as count_i).
    """

    component_keys: list[str]  # Flattened list: ["h.0.attn.q_proj:0", "h.0.attn.q_proj:1", ...]
    count_i: Float[Tensor, " n_components"]  # Firing count per component
    count_ij: Float[Tensor, "n_components n_components"]  # Co-occurrence (upper triangular)
    n_tokens: int  # Total tokens seen

    def _key_to_idx(self, key: str) -> int:
        return self.component_keys.index(key)

    def _get_cooccurrence_row(self, i: int) -> Float[Tensor, " n_components"]:
        """Get full co-occurrence row for component i, reconstructing from upper triangular."""
        # Upper triangular: row i has values at columns >= i
        # We need to also get values from column i in rows < i
        row = self.count_ij[i, :].clone()  # Get row (has values for j >= i)
        row[:i] = self.count_ij[:i, i]  # Fill in values for j < i from column i
        return row

    def get_correlated(
        self,
        component_key: str,
        metric: Metric,
        top_k: int = 10,
    ) -> list[CorrelatedComponent]:
        """Get top-k correlated components for a given component (vectorized).

        Args:
            component_key: The component to find correlations for (e.g., "h.0.attn.q_proj:5")
            metric: Which correlation metric to use
            top_k: Number of top correlations to return

        Returns:
            List of CorrelatedComponent sorted by score descending
        """
        i = self._key_to_idx(component_key)
        count_i_val = self.count_i[i]

        if count_i_val == 0:
            return []

        # Get vectorized co-occurrence counts for component i
        count_ij_row = self._get_cooccurrence_row(i)

        # Compute scores vectorized
        scores = self._compute_metric_vectorized(metric, count_i_val, self.count_i, count_ij_row)

        # Mask out self and zeros
        scores[i] = float("-inf")
        scores[self.count_i == 0] = float("-inf")
        scores[count_ij_row == 0] = float("-inf")

        # Get top-k indices
        top_k_clamped = min(top_k, len(scores))
        top_values, top_indices = torch.topk(scores, top_k_clamped)

        # Filter out -inf values and build result
        result = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            if val == float("-inf"):
                break
            result.append(CorrelatedComponent(component_key=self.component_keys[idx], score=val))

        return result

    def _compute_metric_vectorized(
        self,
        metric: Metric,
        count_i: Tensor,
        count_j: Float[Tensor, " n"],
        count_ij: Float[Tensor, " n"],
    ) -> Float[Tensor, " n"]:
        """Compute correlation metric for all j given fixed i (vectorized)."""
        # Avoid division by zero
        eps = 1e-10

        if metric == "precision":
            # P(j | i) = count_ij / count_i
            return count_ij / (count_i + eps)
        elif metric == "recall":
            # P(i | j) = count_ij / count_j
            return count_ij / (count_j + eps)
        elif metric == "f1":
            precision = count_ij / (count_i + eps)
            recall = count_ij / (count_j + eps)
            return 2 * precision * recall / (precision + recall + eps)
        elif metric == "jaccard":
            union = count_i + count_j - count_ij
            return count_ij / (union + eps)
        elif metric == "pmi":
            # PMI = log(P(i,j) / (P(i) * P(j))) = log(count_ij * n / (count_i * count_j))
            # Use log(a/b) = log(a) - log(b) for numerical stability
            log_joint = torch.log(count_ij * self.n_tokens + eps)
            log_marginal = torch.log(count_i + eps) + torch.log(count_j + eps)
            return log_joint - log_marginal
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save(self, path: Path) -> None:
        """Save correlations to a .pt file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "count_i": self.count_i.cpu(),
                "count_ij": self.count_ij.cpu(),
                "n_tokens": self.n_tokens,
            },
            path,
        )
        logger.info(f"Saved component correlations to {path}")

    @classmethod
    def load(cls, path: Path) -> "ComponentCorrelations":
        """Load correlations from a .pt file."""
        data = torch.load(path, weights_only=True)
        return cls(
            component_keys=data["component_keys"],
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            n_tokens=data["n_tokens"],
        )


def get_correlations_path(wandb_run_id: str) -> Path:
    """Get the path where correlations are stored for a run."""
    return CORRELATIONS_DIR / wandb_run_id / "correlations.pt"


def harvest_correlations(
    config: Config,
    cm: ComponentModel,
    train_loader: DataLoader[Int[Tensor, "B S"]],
    ci_threshold: float,
    n_batches: int,
) -> ComponentCorrelations:
    """Stream through dataset, accumulate co-occurrence counts, compute correlations.

    Optimized for GPU execution with:
    - Half precision for binary activations (reduces memory bandwidth)
    - Fused operations where possible
    - Minimal CPU-GPU transfers

    Args:
        config: Model config (for sampling settings)
        cm: Component model to compute CI values
        train_loader: DataLoader yielding batches of token IDs
        ci_threshold: Threshold for binarizing CI values (component is "active" if CI > threshold)
        n_batches: Number of batches to process

    Returns:
        ComponentCorrelations with accumulated statistics
    """
    device = next(cm.parameters()).device

    # Build flattened component key list: ["layer:cIdx", ...]
    component_keys: list[str] = []
    layer_names = list(cm.target_module_paths)
    for layer_name in layer_names:
        for c_idx in range(cm.C):
            component_keys.append(f"{layer_name}:{c_idx}")

    n_components = len(component_keys)
    n_layers = len(layer_names)
    logger.info(
        f"Harvesting correlations for {n_components} components "
        f"({n_layers} layers × {cm.C} components)"
    )

    # Accumulators - use float32 for accumulation stability
    count_i = torch.zeros(n_components, device=device, dtype=torch.float32)
    count_ij = torch.zeros(n_components, n_components, device=device, dtype=torch.float32)
    n_tokens = 0

    # Pre-allocate reusable tensors to avoid repeated allocation
    # We'll resize if needed but this helps with typical batch sizes

    train_iter = iter(train_loader)
    for _ in tqdm.tqdm(range(n_batches), desc="Harvesting correlations"):
        batch: Int[Tensor, "B S"] = extract_batch_data(next(train_iter)).to(device, non_blocking=True)
        B, S = batch.shape
        n_tokens += B * S

        with torch.no_grad():
            output_with_cache = cm(batch, cache_type="input")
            ci_vals = cm.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                detach_inputs=True,
                sampling=config.sampling,
            ).lower_leaky

            # Stack CI values across layers: (B, S, n_layers, C) -> (B, S, n_components)
            # Use torch.cat with pre-allocated list for slightly better perf than stack
            ci_list = [ci_vals[layer] for layer in layer_names]
            ci_stacked = torch.stack(ci_list, dim=2)
            ci_flat: Float[Tensor, "B S n_components"] = ci_stacked.view(B, S, n_components)

            # Binarize - use half precision for the matmul (2x memory bandwidth savings)
            # The threshold comparison produces bool, convert to float16 for matmul
            binary_acts = (ci_flat > ci_threshold).to(torch.float16)

            # Accumulate firing counts (sum to float32)
            count_i += binary_acts.sum(dim=(0, 1), dtype=torch.float32)

            # Accumulate co-occurrence: (B*S, n_components).T @ (B*S, n_components)
            # This is the hot path - matmul benefits from tensor cores with float16
            flat = binary_acts.view(B * S, n_components)
            # matmul in float16, accumulate to float32
            count_ij += (flat.T @ flat).to(torch.float32)

    # Keep only upper triangular (including diagonal)
    count_ij = torch.triu(count_ij)

    return ComponentCorrelations(
        component_keys=component_keys,
        count_i=count_i,
        count_ij=count_ij,
        n_tokens=n_tokens,
    )
