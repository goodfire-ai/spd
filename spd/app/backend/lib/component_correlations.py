"""Component correlation computation and storage.

Computes pairwise correlation metrics between components based on their
co-occurrence patterns across a dataset. Metrics include precision, recall,
F1, Jaccard similarity, and PMI.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.app.backend.db.database import CORRELATIONS_DIR
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


class MetricFn(Protocol):
    def __call__(
        self,
        count_i: Tensor,
        count_j: Float[Tensor, " n"],
        count_ij: Float[Tensor, " n"],
        n_tokens: int,
    ) -> Float[Tensor, " n"]: ...


def _precision(
    count_i: Tensor,
    count_j: Float[Tensor, " n"],  # pyright: ignore[reportUnusedParameter]
    count_ij: Float[Tensor, " n"],
    n_tokens: int,  # pyright: ignore[reportUnusedParameter]
) -> Float[Tensor, " n"]:
    return torch.where(count_i > 0, count_ij / count_i, float("-inf"))


def _recall(
    count_i: Tensor,  # pyright: ignore[reportUnusedParameter]
    count_j: Float[Tensor, " n"],
    count_ij: Float[Tensor, " n"],
    n_tokens: int,  # pyright: ignore[reportUnusedParameter]
) -> Float[Tensor, " n"]:
    return torch.where(count_j > 0, count_ij / count_j, float("-inf"))


def _f1(
    count_i: Tensor,
    count_j: Float[Tensor, " n"],
    count_ij: Float[Tensor, " n"],
    n_tokens: int,
) -> Float[Tensor, " n"]:
    precision = _precision(count_i, count_j, count_ij, n_tokens)
    recall = _recall(count_i, count_j, count_ij, n_tokens)
    denominator = precision + recall
    return torch.where(denominator > 0, 2 * precision * recall / denominator, float("-inf"))


def _jaccard(
    count_i: Tensor,
    count_j: Float[Tensor, " n"],
    count_ij: Float[Tensor, " n"],
    n_tokens: int,  # pyright: ignore[reportUnusedParameter]
) -> Float[Tensor, " n"]:
    union = count_i + count_j - count_ij
    return torch.where(union > 0, count_ij / union, float("-inf"))


def _pmi(
    count_i: Tensor,
    count_j: Float[Tensor, " n"],
    count_ij: Float[Tensor, " n"],
    n_tokens: int,
) -> Float[Tensor, " n"]:
    return torch.where(
        count_ij > 0.0, math.log(count_ij * n_tokens / (count_i * count_j)), float("-inf")
    )


Metric = Literal["precision", "recall", "f1", "jaccard", "pmi"]
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

    def _compute_scores(
        self, component_key: str, metric: Metric
    ) -> tuple[Tensor, int] | None:
        """Compute scores for a component. Returns (scores, component_idx) or None if no firings."""
        i = self._key_to_idx(component_key)
        count_i_val = self.count_i[i]

        if count_i_val == 0:
            return None

        count_ij_row = self._get_cooccurrence_row(i)
        scores = METRIC_FNS[metric](count_i_val, self.count_i, count_ij_row, self.n_tokens)

        # Mask out self and zeros with -inf
        scores[i] = float("-inf")
        scores[self.count_i == 0] = float("-inf")
        scores[count_ij_row == 0] = float("-inf")

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
        """Load correlations from a .pt file using memory mapping."""
        data = torch.load(path, weights_only=True, mmap=True)
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
        batch: Int[Tensor, "B S"] = extract_batch_data(next(train_iter)).to(
            device, non_blocking=True
        )
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

            # Binarize to float32
            binary_acts = (ci_flat > ci_threshold).to(torch.float32)

            # Accumulate firing counts
            count_i += binary_acts.sum(dim=(0, 1))

            # Accumulate co-occurrence: (B*S, n_components).T @ (B*S, n_components)
            flat = binary_acts.view(B * S, n_components)
            count_ij += flat.T @ flat

            assert torch.isfinite(count_i).all(), "count_i has non-finite values"
            assert torch.isfinite(count_ij).all(), "count_ij has non-finite values"

    # Keep only upper triangular (including diagonal)
    count_ij = torch.triu(count_ij)

    return ComponentCorrelations(
        component_keys=component_keys,
        count_i=count_i,
        count_ij=count_ij,
        n_tokens=n_tokens,
    )
