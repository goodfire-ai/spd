"""Backwards-compatible re-exports from spd.topology.

ModelAdapter is now TransformerTopology. build_model_adapter is now TransformerTopology().
"""

from spd.models.component_model import ComponentModel
from spd.topology import TransformerTopology

ModelAdapter = TransformerTopology


def build_model_adapter(model: ComponentModel) -> TransformerTopology:
    """Build a TransformerTopology. Prefer TransformerTopology(model) directly."""
    return TransformerTopology(model)
