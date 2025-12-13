"""Activation contexts endpoints."""

import json
import queue
import threading
from collections.abc import Generator
from typing import Annotated, Any

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from torch.utils.data import DataLoader

from spd.app.backend.compute import compute_ci_only
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.lib.activation_contexts import get_activations_data
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    ComponentExplanationResponse,
    ComponentExplanationUpdate,
    SubcomponentActivationContexts,
    SubcomponentMetadata,
)
from spd.app.backend.utils import log_errors
from spd.log import logger
from spd.utils.distributed_utils import get_device


class HarvestMetadata(BaseModel):
    """Lightweight metadata returned after harvest, containing only indices and mean_ci values"""

    layers: dict[str, list[SubcomponentMetadata]]


class ComponentProbeRequest(BaseModel):
    """Request to probe a component's CI on custom text."""

    text: str
    layer: str
    component_idx: int


class ComponentProbeResponse(BaseModel):
    """Response with CI values for a component on custom text."""

    tokens: list[str]
    ci_values: list[float]


class CorrelatedComponent(BaseModel):
    """A component correlated with a query component."""

    component_key: str
    score: float


class ComponentCorrelationsResponse(BaseModel):
    """Correlation data for a component across different metrics."""

    precision: list[CorrelatedComponent]
    recall: list[CorrelatedComponent]
    jaccard: list[CorrelatedComponent]
    pmi: list[CorrelatedComponent]


class TokenPRLiftPMI(BaseModel):
    """Token precision, recall, lift, and PMI lists."""

    top_recall: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_precision: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_lift: list[tuple[str, float]]  # [(token, lift), ...] sorted desc
    top_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest positive association
    bottom_pmi: list[tuple[str, float]] | None  # [(token, pmi), ...] highest negative association


class TokenStatsResponse(BaseModel):
    """Token stats for a component (from batch job).

    Contains both input token stats (what tokens activate this component)
    and output token stats (what tokens this component predicts).
    """

    input: TokenPRLiftPMI  # Stats for input tokens
    output: TokenPRLiftPMI  # Stats for output (predicted) tokens


router = APIRouter(prefix="/api/activation_contexts", tags=["activation_contexts"])


@router.get("/summary")
@log_errors
def get_activation_contexts_summary(
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> dict[str, list[SubcomponentMetadata]]:
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    summary = manager.db.get_component_activation_contexts_summary(
        loaded.run.id, loaded.context_length
    )
    if summary is None:
        logger.error(
            f"No activation contexts found for {loaded.run.wandb_path} at context length {loaded.context_length}"
        )
        raise HTTPException(
            status_code=404, detail="No activation contexts found. Generate them first."
        )
    return summary


@router.get("/config")
@log_errors
def get_activation_contexts_config(
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> ActivationContextsGenerationConfig | None:
    """Return the config used to generate the stored activation contexts."""
    return manager.db.get_component_activation_contexts_config(loaded.run.id, loaded.context_length)


@router.get("/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(
    layer: str,
    component_idx: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> SubcomponentActivationContexts:
    """Return full activation context data for a single component."""
    detail = manager.db.get_component_activation_context_detail(
        loaded.run.id, loaded.context_length, layer, component_idx
    )
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Component {layer}:{component_idx} not found")

    # Load explanation from database
    component_key = f"{layer}:{component_idx}"
    explanation = manager.db.get_component_explanation(
        loaded.run.id, loaded.context_length, component_key
    )
    detail.explanation = explanation
    return detail


@router.get("/subcomponents")
@log_errors
def generate_activation_contexts(
    importance_threshold: Annotated[float, Query(ge=0, le=1)],
    n_batches: Annotated[int, Query(ge=1)],
    batch_size: Annotated[int, Query(ge=1)],
    n_tokens_either_side: Annotated[int, Query(ge=0)],
    topk_examples: Annotated[int, Query(ge=1)],
    separation_tokens: Annotated[int, Query(ge=0)],
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> StreamingResponse:
    """Generate activation contexts from training data.

    This streams progress updates and saves the result to DB when complete.
    """
    assert separation_tokens <= n_tokens_either_side, (
        "separation_tokens must be less than or equal to n_tokens_either_side"
    )
    db = manager.db

    # Create a data loader with user-specified batch size using the existing dataset
    train_loader = DataLoader(
        loaded.train_loader.dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset already shuffled, don't double-shuffle
    )

    config = ActivationContextsGenerationConfig(
        importance_threshold=importance_threshold,
        n_batches=n_batches,
        batch_size=batch_size,
        n_tokens_either_side=n_tokens_either_side,
        topk_examples=topk_examples,
        separation_tokens=separation_tokens,
    )

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(progress: float) -> None:
        progress_queue.put({"type": "progress", "progress": progress})

    def compute_thread() -> None:
        try:
            act_contexts = get_activations_data(
                config=loaded.config,
                cm=loaded.model,
                tokenizer=loaded.tokenizer,
                train_loader=train_loader,
                token_strings=loaded.token_strings,
                importance_threshold=importance_threshold,
                n_batches=n_batches,
                n_tokens_either_side=n_tokens_either_side,
                topk_examples=topk_examples,
                separation_tokens=separation_tokens,
                onprogress=on_progress,
            )
            logger.info("Saving activation contexts to database...")
            db.set_component_activation_contexts(
                loaded.run.id, loaded.context_length, act_contexts, config
            )
            logger.info("Saved activation contexts to database")

            metadata = HarvestMetadata(
                layers={
                    layer_name: [
                        SubcomponentMetadata(
                            subcomponent_idx=subcomp.subcomponent_idx,
                            mean_ci=subcomp.mean_ci,
                        )
                        for subcomp in subcomponents
                    ]
                    for layer_name, subcomponents in act_contexts.layers.items()
                },
            )
            progress_queue.put({"type": "complete", "result": metadata.model_dump()})
        except Exception as e:
            progress_queue.put({"type": "error", "error": str(e)})

    def generate() -> Generator[str]:
        thread = threading.Thread(target=compute_thread)
        thread.start()

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if msg["type"] == "progress":
                yield f"data: {json.dumps(msg)}\n\n"
            elif msg["type"] == "error" or msg["type"] == "complete":
                yield f"data: {json.dumps(msg)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/probe")
@log_errors
def probe_component(
    request: ComponentProbeRequest,
    loaded: DepLoadedRun,
) -> ComponentProbeResponse:
    """Probe a component's CI values on custom text.

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

    ci_tensor = result.ci_lower_leaky[request.layer]
    ci_values = ci_tensor[0, :, request.component_idx].tolist()
    token_strings = [loaded.token_strings[t] for t in token_ids]

    return ComponentProbeResponse(tokens=token_strings, ci_values=ci_values)


@router.put("/{layer}/{component_idx}/explanation")
@log_errors
def set_component_explanation(
    layer: str,
    component_idx: int,
    body: ComponentExplanationUpdate,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> ComponentExplanationResponse:
    """Set or update the explanation for a component."""
    component_key = f"{layer}:{component_idx}"
    manager.db.set_component_explanation(
        loaded.run.id, loaded.context_length, component_key, body.explanation
    )
    return ComponentExplanationResponse(component_key=component_key, explanation=body.explanation)


@router.get("/{layer}/{component_idx}/explanation")
@log_errors
def get_component_explanation(
    layer: str,
    component_idx: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> ComponentExplanationResponse:
    """Get the explanation for a component."""
    component_key = f"{layer}:{component_idx}"
    explanation = manager.db.get_component_explanation(
        loaded.run.id, loaded.context_length, component_key
    )
    return ComponentExplanationResponse(component_key=component_key, explanation=explanation)


@router.delete("/{layer}/{component_idx}/explanation")
@log_errors
def delete_component_explanation(
    layer: str,
    component_idx: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> dict[str, str]:
    """Delete the explanation for a component."""
    component_key = f"{layer}:{component_idx}"
    manager.db.delete_component_explanation(loaded.run.id, loaded.context_length, component_key)
    return {"status": "deleted"}
