"""Thin SAE integration adapters for the generic decomposition pipeline."""

from typing import Protocol

from jaxtyping import Float
from torch import Tensor, nn

from spd.decomp.types import ActivationFn, BatchLike

from .common import per_feature_dict, run_model, tensor_from_output


class SAEOutputProtocol(Protocol):
    """Minimal expected SAE output shape: per-token latent activations [B, S, F]."""

    latents: Float[Tensor, "batch seq feature"]


def make_sae_activation_fn(
    latent_key: str = "latents",
    feature_prefix: str = "sae",
    take_abs: bool = True,
) -> ActivationFn:
    """Build ActivationFn for SAE-style per-token latent activations."""

    def _activation_fn(
        model: nn.Module,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]:
        out = run_model(model, batch)
        latents: Float[Tensor, "batch seq feature"] = tensor_from_output(out, latent_key)
        if take_abs:
            latents = latents.abs()
        return per_feature_dict(latents, feature_prefix)

    return _activation_fn
