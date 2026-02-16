"""Shared helpers for method adapter stubs (SAE/CLT/MOLT)."""

from jaxtyping import Float
from torch import Tensor, nn

from spd.decomp.types import BatchLike


def run_model(model: nn.Module, batch: BatchLike) -> object:
    """Run model on a batch payload in common dict/tuple/tensor formats."""
    if isinstance(batch, dict):
        try:
            return model(**batch)
        except TypeError:
            return model(batch)
    if isinstance(batch, tuple):
        try:
            return model(*batch)
        except TypeError:
            return model(batch)
    return model(batch)


def tensor_from_output(output: object, tensor_key: str) -> Tensor:
    """Extract a feature tensor from common model output containers."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict):
        value = output.get(tensor_key)
        if isinstance(value, Tensor):
            return value
    value = getattr(output, tensor_key, None)
    if isinstance(value, Tensor):
        return value
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise KeyError(f"Could not find tensor key '{tensor_key}' in model output")


def per_feature_dict(
    features: Float[Tensor, "batch seq feature"],
    prefix: str,
) -> dict[str, Float[Tensor, "batch seq"]]:
    """Split [B, S, F] feature activations into component-key dict form."""
    assert features.ndim == 3, f"Expected [B, S, F], got shape={tuple(features.shape)}"
    return {f"{prefix}:{i}": features[:, :, i] for i in range(features.shape[2])}
