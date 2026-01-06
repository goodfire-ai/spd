"""Activation contexts endpoints.

These endpoints serve activation context data from the harvest pipeline output.
"""

from collections import defaultdict

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.compute import compute_ci_only
from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.schemas import SubcomponentActivationContexts, SubcomponentMetadata
from spd.app.backend.utils import log_errors
from spd.utils.distributed_utils import get_device


class ComponentProbeRequest(BaseModel):
    """Request to probe a component's CI on custom text."""

    text: str
    layer: str
    component_idx: int


class ComponentProbeResponse(BaseModel):
    """Response with CI and subcomponent activation values for a component on custom text."""

    tokens: list[str]
    ci_values: list[float]
    subcomp_acts: list[float]


router = APIRouter(prefix="/api/activation_contexts", tags=["activation_contexts"])


@router.get("/summary")
@log_errors
def get_activation_contexts_summary(
    loaded: DepLoadedRun,
) -> dict[str, list[SubcomponentMetadata]]:
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    contexts = loaded.harvest.activation_contexts
    if contexts is None:
        raise HTTPException(
            status_code=404,
            detail="No activation contexts found. Run harvest first.",
        )

    # Group by layer
    summary: dict[str, list[SubcomponentMetadata]] = defaultdict(list)
    for comp in contexts.values():
        # Get mean subcomponent activation if available (new harvests)
        summary[comp.layer].append(
            SubcomponentMetadata(
                subcomponent_idx=comp.component_idx,
                mean_ci=comp.mean_ci,
                mean_subcomp_act=comp.subcomp_act_stats.mean,
            )
        )

    # Sort by mean CI descending within each layer
    for layer in summary:
        summary[layer].sort(key=lambda x: x.mean_ci, reverse=True)

    return dict(summary)


@router.get("/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> SubcomponentActivationContexts:
    """Return full activation context data for a single component."""
    contexts = loaded.harvest.activation_contexts
    if contexts is None:
        raise HTTPException(
            status_code=404,
            detail="No activation contexts found. Run harvest first.",
        )

    component_key = f"{layer}:{component_idx}"
    comp = contexts.get(component_key)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component {component_key} not found")

    # Convert token IDs to strings
    PADDING_SENTINEL = -1
    token_strings = loaded.token_strings

    def token_str(tid: int) -> str:
        if tid == PADDING_SENTINEL:
            return "<pad>"
        assert tid in token_strings, f"Token ID {tid} not in vocab"
        return token_strings[tid]

    example_tokens = [[token_str(tid) for tid in ex.token_ids] for ex in comp.activation_examples]
    example_ci = [ex.ci_values for ex in comp.activation_examples]
    example_inner_acts = [ex.inner_acts for ex in comp.activation_examples]

    return SubcomponentActivationContexts(
        subcomponent_idx=comp.component_idx,
        mean_ci=comp.mean_ci,
        mean_subcomp_act=comp.subcomp_act_stats.mean,
        example_tokens=example_tokens,
        example_ci=example_ci,
        example_inner_acts=example_inner_acts,
    )


@router.post("/probe")
@log_errors
def probe_component(
    request: ComponentProbeRequest,
    loaded: DepLoadedRun,
) -> ComponentProbeResponse:
    """Probe a component's CI and subcomponent activation values on custom text.

    Fast endpoint for testing hypotheses about component activation.
    Only requires a single forward pass.
    """
    device = get_device()

    token_ids = loaded.tokenizer.encode(request.text, add_special_tokens=False)
    assert len(token_ids) > 0, "Text produced no tokens"

    tokens_tensor = torch.tensor([token_ids], device=device)

    result = compute_ci_only(
        model=loaded.model,
        tokens=tokens_tensor,
        sampling=loaded.config.sampling,
    )

    assert request.layer in loaded.model.components, f"Layer {request.layer} not in model"

    ci_tensor = result.ci_lower_leaky[request.layer]
    ci_values = ci_tensor[0, :, request.component_idx].tolist()
    token_strings = [loaded.token_strings[t] for t in token_ids]

    subcomp_acts_tensor = result.component_acts[request.layer]
    subcomp_acts = subcomp_acts_tensor[0, :, request.component_idx].tolist()

    return ComponentProbeResponse(
        tokens=token_strings, ci_values=ci_values, subcomp_acts=subcomp_acts
    )
