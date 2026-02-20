"""Harvest method adapters: method-specific logic for the generic harvest pipeline.

Each decomposition method (SPD, CLT, MOLT) provides an adapter that knows how to:
- Load the model and build a dataloader
- Compute firings and activations from a batch (harvest_fn)
- Report layer structure and vocab size

Construct via adapter_from_config(method_config).
"""

from spd.adapters.base import DecompositionAdapter


def adapter_from_id(id: str) -> DecompositionAdapter:
    from spd.adapters.spd import SPDAdapter

    if id.startswith("s-"):
        return SPDAdapter(id)
    elif id.startswith("clt-"):
        raise NotImplementedError("CLT adapter not implemented yet")
    elif id.startswith("molt-"):
        raise NotImplementedError("MOLT adapter not implemented yet")

    raise ValueError(f"Unsupported decomposition ID: {id}")
