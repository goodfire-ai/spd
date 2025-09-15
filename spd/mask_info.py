from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied to a ComponentOrModule object."""

    module_name: str
    mask: Float[Tensor, "... C"]
    weight_delta: Float[Tensor, " d_out d_in"] | None
    weight_delta_mask: Float[Tensor, "..."] | None


def make_mask_infos(
    masks: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None = None,
    weight_delta_masks: dict[str, Float[Tensor, "..."]] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Create ComponentsMaskInfo dict from dicts of masks, weight deltas, and weight delta masks.

    If weight_deltas or weight_delta_masks are provided, their keys must match masks.keys().

    Args:
        masks: Dict of masks to be applied to the components of a ComponentOrModule object.
        weight_deltas: Dict of weight deltas for each module to be decomposed.
        weight_delta_masks: Dict of masks to be applied to the weight deltas of each module.
    Returns:
        Dict mapping module names to ComponentsMaskInfo objects.
    """
    if weight_deltas is not None:
        assert set(weight_deltas) == set(masks)
    if weight_delta_masks is not None:
        assert weight_deltas is not None
        assert set(weight_delta_masks) == set(masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in masks:
        result[name] = ComponentsMaskInfo(
            module_name=name,
            mask=masks[name],
            weight_delta=None if weight_deltas is None else weight_deltas[name],
            weight_delta_mask=None if weight_delta_masks is None else weight_delta_masks[name],
        )

    return result
