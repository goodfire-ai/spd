"""Evaluation through the CI lifecycle interface."""

from jaxtyping import Array

from param_decomp.core.ci_fn.interface import CI, CIFn
from param_decomp.core.model import CaptureRequest, Captures, ComponentActivations
from param_decomp.sequence import SequenceLayout


def evaluate_ci_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: Captures,
    conditioning: Conditioning,
    components: ComponentActivations,
    *,
    sequence: SequenceLayout | None,
    remat: bool,
) -> CI:
    """Evaluate from the physical target captures CI requested, read as activation arrays;
    `captures` may hold others, which CI never sees."""
    return ci_fn(
        captures.read(CaptureRequest(ci_fn.capture_keys, Array)),
        conditioning,
        components,
        sequence=sequence,
        remat=remat,
    )
