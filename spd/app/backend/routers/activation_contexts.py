"""Activation contexts endpoints."""

import json
import queue
import threading
import time
from collections.abc import Generator
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from torch.utils.data import DataLoader

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.lib.activation_contexts import get_activations_data
from spd.app.backend.lib.component_correlations import (
    ComponentCorrelations,
    ComponentTokenStats,
    get_correlations_path,
    get_token_stats_path,
)
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    ComponentCorrelationsResponse,
    ComponentProbeRequest,
    ComponentProbeResponse,
    CorrelatedComponent,
    HarvestMetadata,
    SubcomponentActivationContexts,
    SubcomponentMetadata,
    TokenPRLiftPMI,
    TokenStatsResponse,
)
from spd.app.backend.utils import log_errors
from spd.log import logger

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
                # TODO: Re-enable token uplift after performance optimization
                # token_base_rates=loaded.token_base_rates,
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
    import torch

    from spd.app.backend.compute import compute_ci_only
    from spd.utils.distributed_utils import get_device

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


def _get_correlations(run_id: str) -> ComponentCorrelations | None:
    """Load correlations from cache or disk."""
    start = time.perf_counter()

    path = get_correlations_path(run_id)
    if not path.exists():
        return None

    correlations = ComponentCorrelations.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Loaded correlations for {run_id} in {load_ms:.1f}ms")
    return correlations


def _get_token_stats(run_id: str) -> ComponentTokenStats | None:
    """Load token stats from cache or disk."""
    start = time.perf_counter()

    path = get_token_stats_path(run_id)
    if not path.exists():
        return None

    token_stats = ComponentTokenStats.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Loaded token stats for {run_id} in {load_ms:.1f}ms")
    return token_stats


@router.get("/correlations/{layer}/{component_idx}")
@log_errors
def get_component_correlations(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> ComponentCorrelationsResponse | None:
    """Get correlated components for a specific component.

    Returns top-k correlations across different metrics (precision, recall, F1, Jaccard).
    Returns None if correlations haven't been harvested for this run.
    """
    start = time.perf_counter()

    # Extract run_id from wandb_path (entity/project/run_id -> run_id)
    run_id = loaded.run.wandb_path.split("/")[-1]
    correlations = _get_correlations(run_id)

    if correlations is None:
        return None

    component_key = f"{layer}:{component_idx}"

    if component_key not in correlations.component_keys:
        raise HTTPException(
            status_code=404, detail=f"Component {component_key} not found in correlations"
        )

    from spd.app.backend.lib.component_correlations import (
        CorrelatedComponent as CorrelatedComponentDC,
    )

    def to_schema(c: CorrelatedComponentDC) -> CorrelatedComponent:
        return CorrelatedComponent(component_key=c.component_key, score=c.score)

    response = ComponentCorrelationsResponse(
        precision=[
            to_schema(c) for c in correlations.get_correlated(component_key, "precision", top_k)
        ],
        recall=[to_schema(c) for c in correlations.get_correlated(component_key, "recall", top_k)],
        f1=[to_schema(c) for c in correlations.get_correlated(component_key, "f1", top_k)],
        jaccard=[
            to_schema(c) for c in correlations.get_correlated(component_key, "jaccard", top_k)
        ],
        pmi=[to_schema(c) for c in correlations.get_correlated(component_key, "pmi", top_k)],
    )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(f"get_component_correlations: {component_key} in {total_ms:.1f}ms")
    return response


@router.get("/token_stats/{layer}/{component_idx}")
@log_errors
def get_component_token_stats(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> TokenStatsResponse | None:
    """Get token precision/recall/lift/PMI for a component.

    Returns stats for both input tokens (what activates this component)
    and output tokens (what this component predicts).
    Returns None if token stats haven't been harvested for this run.
    """
    start = time.perf_counter()

    run_id = loaded.run.wandb_path.split("/")[-1]
    token_stats = _get_token_stats(run_id)

    if token_stats is None:
        return None

    component_key = f"{layer}:{component_idx}"

    input_stats = token_stats.get_input_stats(component_key, loaded.tokenizer, top_k=top_k)
    output_stats = token_stats.get_output_stats(component_key, loaded.tokenizer, top_k=top_k)

    if input_stats is None or output_stats is None:
        return None

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(f"get_component_token_stats: {component_key} in {total_ms:.1f}ms")

    return TokenStatsResponse(
        input=TokenPRLiftPMI(
            top_recall=input_stats.top_recall,
            top_precision=input_stats.top_precision,
            top_lift=input_stats.top_lift,
            top_pmi=input_stats.top_pmi,
            bottom_pmi=input_stats.bottom_pmi,
        ),
        output=TokenPRLiftPMI(
            top_recall=output_stats.top_recall,
            top_precision=output_stats.top_precision,
            top_lift=output_stats.top_lift,
            top_pmi=output_stats.top_pmi,
            bottom_pmi=output_stats.bottom_pmi,
        ),
    )
