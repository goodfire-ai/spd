from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

WeightDeltaAndMask = tuple[Float[Tensor, " d_out d_in"], Float[Tensor, "..."]]


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied to a ComponentOrModule object."""

    module_name: str
    component_mask: Float[Tensor, "... C"]
    weight_delta_and_mask: WeightDeltaAndMask | None


def make_mask_infos(
    masks: dict[str, Float[Tensor, "... C"]],
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Create ComponentsMaskInfo dict from dicts of masks, weight deltas, and weight delta masks.

    If weight_deltas_and_masks is provided, it's keys must match masks.keys().

    Args:
        masks: Dict of masks to be applied to the components of a ComponentOrModule object.
        weight_deltas_and_masks: Dict of weight deltas and masks for each module to be decomposed.
    turns:
        Dict mapping module names to ComponentsMaskInfo objects.
    """
    if weight_deltas_and_masks is not None:
        assert set(weight_deltas_and_masks) == set(masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in masks:
        result[name] = ComponentsMaskInfo(
            module_name=name,
            component_mask=masks[name],
            weight_delta_and_mask=None
            if weight_deltas_and_masks is None
            else weight_deltas_and_masks[name],
        )

    return result
