"""Activation contexts endpoints."""

import json
import queue
import threading
from collections.abc import Generator
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from torch.utils.data import DataLoader

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.lib.activation_contexts import get_activations_data
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    HarvestMetadata,
    ModelActivationContexts,
    SubcomponentActivationContexts,
    SubcomponentMetadata,
)
from spd.app.backend.state import RunState, StateManager
from spd.app.backend.utils import log_errors

router = APIRouter(prefix="/api/activation_contexts", tags=["activation_contexts"])


def _ensure_activation_contexts_cached(
    manager: StateManager,
    loaded: RunState,
) -> ModelActivationContexts | None:
    """Load activation contexts into cache if not already loaded."""
    if loaded.activation_contexts_cache is None:
        contexts = manager.db.get_activation_contexts(loaded.run.id)
        if contexts is not None:
            loaded.activation_contexts_cache = contexts
    return loaded.activation_contexts_cache


@router.get("/summary")
@log_errors
def get_activation_contexts_summary(
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> dict[str, list[SubcomponentMetadata]]:
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    contexts = _ensure_activation_contexts_cached(manager, loaded)

    if contexts is None:
        raise HTTPException(
            status_code=404, detail="No activation contexts found. Generate them first."
        )

    summary: dict[str, list[SubcomponentMetadata]] = {}
    for layer, subcomps in contexts.layers.items():
        summary[layer] = [
            SubcomponentMetadata(subcomponent_idx=s.subcomponent_idx, mean_ci=s.mean_ci)
            for s in subcomps
        ]
    return summary


@router.get("/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(
    layer: str,
    component_idx: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> SubcomponentActivationContexts:
    """Return full activation context data for a single component."""
    contexts = _ensure_activation_contexts_cached(manager, loaded)

    if contexts is None:
        raise HTTPException(status_code=404, detail="No activation contexts found")

    layer_data = contexts.layers.get(layer)
    if layer_data is None:
        raise HTTPException(status_code=404, detail=f"Layer '{layer}' not found")

    for subcomp in layer_data:
        if subcomp.subcomponent_idx == component_idx:
            return subcomp

    raise HTTPException(
        status_code=404, detail=f"Component {component_idx} not found in layer '{layer}'"
    )


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
            db.set_activation_contexts(loaded.run.id, act_contexts, config)

            # Clear cache so it reloads from DB
            loaded.activation_contexts_cache = None

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
