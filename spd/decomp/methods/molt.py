"""Thin MOLT integration adapters for the generic decomposition pipeline."""

from typing import Protocol

from jaxtyping import Float
from torch import Tensor, nn

from spd.decomp.types import ActivationFn, BatchLike

from .common import per_feature_dict, run_model, tensor_from_output


class MOLTOutputProtocol(Protocol):
    """Minimal expected MOLT output shape: per-token feature activations [B, S, F]."""

    molt_features: Float[Tensor, "batch seq feature"]


def make_molt_activation_fn(
    feature_key: str = "molt_features",
    feature_prefix: str = "molt",
    take_abs: bool = False,
) -> ActivationFn:
    """Build ActivationFn for MOLT-style per-token feature activations."""

    def _activation_fn(
        model: nn.Module,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]:
        out = run_model(model, batch)
        features: Float[Tensor, "batch seq feature"] = tensor_from_output(out, feature_key)
        if take_abs:
            features = features.abs()
        return per_feature_dict(features, feature_prefix)

    return _activation_fn
