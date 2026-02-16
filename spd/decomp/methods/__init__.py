"""Method-specific activation fns for generic decomposition pipeline."""

from .spd import make_spd_activation_fn, make_spd_component_acts_fn, spd_output_probs_fn

__all__ = [
    "make_spd_activation_fn",
    "make_spd_component_acts_fn",
    "spd_output_probs_fn",
]
