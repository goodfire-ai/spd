"""Graph computation endpoints for tokenization and attribution graphs."""

import json
import math
import queue
import threading
from collections.abc import Generator
from itertools import groupby
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from spd.app.backend.compute import (
    Edge,
    LocalAttributionResult,
    Node,
    OptimizedLocalAttributionResult,
    compute_local_attributions,
    compute_local_attributions_optimized,
)
from spd.app.backend.db.database import (
    CachedEdge,
    CachedOutputProb,
    OptimizationParams,
    OptimizationStats,
)
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.optim_cis.run_optim_cis import OptimCIConfig
from spd.app.backend.schemas import (
    EdgeData,
    GraphData,
    GraphDataWithOptimization,
    OptimizationResult,
    OutputProbability,
    TokenizeResponse,
)
from spd.app.backend.utils import log_errors
from spd.configs import ImportanceMinimalityLossConfig
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

router = APIRouter(prefix="/api/graphs", tags=["graphs"])

DEVICE = get_device()

# This is a bit of a hack. We want to limit the number of edges returned to avoid overwhelming the frontend.
GLOBAL_EDGE_LIMIT = 5_000


def _parse_node(s: str) -> Node:
    """Parse node string 'layer:seq_pos:component_idx' back to a Node object."""
    parts = s.rsplit(":", 2)
    return Node(layer=parts[0], seq_pos=int(parts[1]), component_idx=int(parts[2]))


@router.post("/tokenize")
@log_errors
def tokenize_text(text: str, loaded: DepLoadedRun) -> TokenizeResponse:
    """Tokenize text and return tokens for preview (special tokens filtered)."""
    token_ids = loaded.tokenizer.encode(text, add_special_tokens=False)

    return TokenizeResponse(
        text=text,
        token_ids=token_ids,
        tokens=[loaded.token_strings[t] for t in token_ids],
    )


NormalizeType = Literal["none", "target", "layer"]


@router.post("")
@log_errors
def compute_graph_stream(
    prompt_id: Annotated[int, Query()],
    normalize: Annotated[NormalizeType, Query()],
    loaded: DepLoadedRun,
    manager: DepStateManager,
):
    """Compute attribution graph for a prompt with streaming progress."""
    ci_threshold = 1e-6
    output_prob_threshold = 0.01

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return JSONResponse({"error": "Prompt not found"}, status_code=404)

    token_ids = prompt.token_ids
    token_strings = [loaded.token_strings[t] for t in token_ids]

    # Check cache first
    cached = db.get_cached_graph(prompt_id, optimization_params=None)
    if cached is not None:
        # Cache hit - return immediately without streaming
        edges = [
            Edge(
                source=_parse_node(e.src),
                target=_parse_node(e.tgt),
                is_cross_seq=e.is_cross_seq,
                strength=e.val,
            )
            for e in cached.edges
        ]

        # Apply normalization
        match normalize:
            case "none":
                pass
            case "target":
                edges = _normalize_edges_by_target(edges)
            case "layer":
                edges = _normalize_edges_by_target_layer(edges)
        if len(edges) > GLOBAL_EDGE_LIMIT:
            edges.sort(key=lambda e: abs(e.strength), reverse=True)
            edges = edges[:GLOBAL_EDGE_LIMIT]

        edges_typed = [EdgeData(src=str(e.source), tgt=str(e.target), val=e.strength) for e in edges]
        output_probs = {k: OutputProbability(prob=v.prob, token=v.token) for k, v in cached.output_probs.items()}

        response_data = GraphData(
            id=prompt_id,
            tokens=token_strings,
            edges=edges_typed,
            outputProbs=output_probs,
            cached=True,
        )

        def generate_cached() -> Generator[str]:
            yield f"data: {json.dumps({'type': 'complete', 'data': response_data.model_dump()})}\n\n"

        return StreamingResponse(generate_cached(), media_type="text/event-stream")

    # Cache miss - compute with streaming progress
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"type": "progress", "current": current, "total": total, "stage": stage})

    def compute_thread() -> None:
        try:
            result = compute_local_attributions(
                model=loaded.model,
                tokens=tokens_tensor,
                sources_by_target=loaded.sources_by_target,
                ci_threshold=ci_threshold,
                output_prob_threshold=output_prob_threshold,
                sampling=loaded.config.sampling,
                device=DEVICE,
                show_progress=False,
                on_progress=on_progress,
            )
            progress_queue.put({"type": "result", "result": result})
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
            elif msg["type"] == "error":
                yield f"data: {json.dumps(msg)}\n\n"
                break
            elif msg["type"] == "result":
                result = runtime_cast(LocalAttributionResult, msg["result"])

                # Store raw edges in cache before normalization
                raw_edges = result.edges
                raw_output_probs_tensor = result.output_probs[0].cpu()
                raw_output_probs: dict[str, CachedOutputProb] = {}
                for s in range(raw_output_probs_tensor.shape[0]):
                    for c_idx in range(raw_output_probs_tensor.shape[1]):
                        prob = float(raw_output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        raw_output_probs[key] = CachedOutputProb(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                # Save to cache (raw, unnormalized edges)
                db.save_cached_graph(
                    prompt_id=prompt_id,
                    edges=[
                        CachedEdge(
                            src=str(e.source),
                            tgt=str(e.target),
                            val=e.strength,
                            is_cross_seq=e.is_cross_seq,
                        )
                        for e in raw_edges
                    ],
                    output_probs=raw_output_probs,
                )

                # Now apply normalization for response
                edges = raw_edges
                match normalize:
                    case "none":
                        pass
                    case "target":
                        edges = _normalize_edges_by_target(edges)
                    case "layer":
                        edges = _normalize_edges_by_target_layer(edges)
                if len(edges) > GLOBAL_EDGE_LIMIT:
                    edges.sort(key=lambda e: abs(e.strength), reverse=True)
                    edges = edges[:GLOBAL_EDGE_LIMIT]

                edges_typed = [
                    EdgeData(src=str(e.source), tgt=str(e.target), val=e.strength) for e in edges
                ]

                output_probs: dict[str, OutputProbability] = {
                    k: OutputProbability(prob=v.prob, token=v.token) for k, v in raw_output_probs.items()
                }

                response_data = GraphData(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
                    cached=False,
                )
                complete_data = {"type": "complete", "data": response_data.model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


def _normalize_edges_by_target(edges: list[Edge]) -> list[Edge]:
    def get_target_node(edge: Edge) -> str:
        return str(edge.target)

    sorted_edges = sorted(edges, key=get_target_node)
    groups = groupby(sorted_edges, key=get_target_node)

    out_edges = []
    for _, incoming_edges in groups:
        incoming_edges = list(incoming_edges)  # list() because we iterate over the group twice
        incoming_strength = math.sqrt(sum(edge.strength**2 for edge in incoming_edges))
        for edge in incoming_edges:
            out_edges.append(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    is_cross_seq=edge.is_cross_seq,
                    strength=edge.strength / incoming_strength,
                )
            )
    return out_edges

def _normalize_edges_by_target_layer(edges: list[Edge]) -> list[Edge]:
    def get_target_layer(edge: Edge) -> str:
        return edge.target.layer

    sorted_edges = sorted(edges, key=get_target_layer)
    groups = groupby(sorted_edges, key=get_target_layer)

    out_edges = []
    for _, incoming_edges in groups:
        incoming_edges = list(incoming_edges)  # list() because we iterate over the group twice
        incoming_strength = math.sqrt(sum(edge.strength**2 for edge in incoming_edges))
        for edge in incoming_edges:
            out_edges.append(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    is_cross_seq=edge.is_cross_seq,
                    strength=edge.strength / incoming_strength,
                )
            )
    return out_edges

@router.post("/optimized/stream")
@log_errors
def compute_graph_optimized_stream(
    prompt_id: Annotated[int, Query()],
    label_token: Annotated[int, Query()],
    imp_min_coeff: Annotated[float, Query(gt=0)],
    ce_loss_coeff: Annotated[float, Query(gt=0)],
    steps: Annotated[int, Query(gt=0)],
    pnorm: Annotated[float, Query(gt=0, le=1)],
    normalize: Annotated[NormalizeType, Query()],
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)],
    loaded: DepLoadedRun,
    manager: DepStateManager,
):
    """Compute optimized attribution graph for a prompt with streaming progress."""
    lr = 1e-2
    ci_threshold = 1e-6

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return JSONResponse({"error": "Prompt not found"}, status_code=404)

    token_ids = prompt.token_ids
    label_str = loaded.token_strings[label_token]
    token_strings = [loaded.token_strings[t] for t in token_ids]

    # Build optimization params for cache key
    opt_params = OptimizationParams(
        label_token=label_token,
        imp_min_coeff=imp_min_coeff,
        ce_loss_coeff=ce_loss_coeff,
        steps=steps,
        pnorm=pnorm,
    )

    # Check cache first
    cached = db.get_cached_graph(prompt_id, optimization_params=opt_params)
    if cached is not None:
        # Cache hit - return immediately without streaming
        edges = [
            Edge(
                source=_parse_node(e.src),
                target=_parse_node(e.tgt),
                is_cross_seq=e.is_cross_seq,
                strength=e.val,
            )
            for e in cached.edges
        ]

        # Apply normalization
        match normalize:
            case "none":
                pass
            case "target":
                edges = _normalize_edges_by_target(edges)
            case "layer":
                edges = _normalize_edges_by_target_layer(edges)
        if len(edges) > GLOBAL_EDGE_LIMIT:
            edges.sort(key=lambda e: abs(e.strength), reverse=True)
            edges = edges[:GLOBAL_EDGE_LIMIT]

        edges_typed = [EdgeData(src=str(e.source), tgt=str(e.target), val=e.strength) for e in edges]
        output_probs = {k: OutputProbability(prob=v.prob, token=v.token) for k, v in cached.output_probs.items()}

        assert cached.optimization_stats is not None
        response_data = GraphDataWithOptimization(
            id=prompt_id,
            tokens=token_strings,
            edges=edges_typed,
            outputProbs=output_probs,
            cached=True,
            optimization=OptimizationResult(
                label_token=label_token,
                label_str=label_str,
                imp_min_coeff=imp_min_coeff,
                ce_loss_coeff=ce_loss_coeff,
                steps=steps,
                label_prob=cached.optimization_stats.label_prob,
                l0_total=cached.optimization_stats.l0_total,
                l0_per_layer=cached.optimization_stats.l0_per_layer,
            ),
        )

        def generate_cached() -> Generator[str]:
            yield f"data: {json.dumps({'type': 'complete', 'data': response_data.model_dump()})}\n\n"

        return StreamingResponse(generate_cached(), media_type="text/event-stream")

    # Cache miss - compute with streaming progress
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    optim_config = OptimCIConfig(
        seed=0,
        lr=lr,
        steps=steps,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, steps // 4),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=imp_min_coeff, pnorm=pnorm),
        ce_loss_coeff=ce_loss_coeff,
        ci_threshold=ci_threshold,
        sampling=loaded.config.sampling,
        ce_kl_rounding_threshold=0.5,
    )

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"type": "progress", "current": current, "total": total, "stage": stage})

    def compute_thread() -> None:
        try:
            result = compute_local_attributions_optimized(
                model=loaded.model,
                tokens=tokens_tensor,
                label_token=label_token,
                sources_by_target=loaded.sources_by_target,
                optim_config=optim_config,
                ci_threshold=ci_threshold,
                output_prob_threshold=output_prob_threshold,
                device=DEVICE,
                show_progress=False,
                on_progress=on_progress,
            )
            progress_queue.put({"type": "result", "result": result})
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
            elif msg["type"] == "error":
                yield f"data: {json.dumps(msg)}\n\n"
                break
            elif msg["type"] == "result":
                result = runtime_cast(OptimizedLocalAttributionResult, msg["result"])

                # Store raw edges in cache before normalization
                raw_edges = result.edges
                raw_output_probs_tensor = result.output_probs[0].cpu()
                raw_output_probs: dict[str, CachedOutputProb] = {}
                for s in range(raw_output_probs_tensor.shape[0]):
                    for c_idx in range(raw_output_probs_tensor.shape[1]):
                        prob = float(raw_output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        raw_output_probs[key] = CachedOutputProb(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                # Save to cache (raw, unnormalized edges)
                db.save_cached_graph(
                    prompt_id=prompt_id,
                    edges=[
                        CachedEdge(
                            src=str(e.source),
                            tgt=str(e.target),
                            val=e.strength,
                            is_cross_seq=e.is_cross_seq,
                        )
                        for e in raw_edges
                    ],
                    output_probs=raw_output_probs,
                    optimization_params=opt_params,
                    optimization_stats=OptimizationStats(
                        label_prob=result.stats.label_prob,
                        l0_total=result.stats.l0_total,
                        l0_per_layer=result.stats.l0_per_layer,
                    ),
                )

                # Now apply normalization for response
                edges = raw_edges
                match normalize:
                    case "none":
                        pass
                    case "target":
                        edges = _normalize_edges_by_target(edges)
                    case "layer":
                        edges = _normalize_edges_by_target_layer(edges)
                if len(edges) > GLOBAL_EDGE_LIMIT:
                    edges.sort(key=lambda e: abs(e.strength), reverse=True)
                    edges = edges[:GLOBAL_EDGE_LIMIT]

                edges_typed = [
                    EdgeData(src=str(e.source), tgt=str(e.target), val=e.strength) for e in edges
                ]

                output_probs: dict[str, OutputProbability] = {
                    k: OutputProbability(prob=v.prob, token=v.token) for k, v in raw_output_probs.items()
                }

                response_data = GraphDataWithOptimization(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
                    cached=False,
                    optimization=OptimizationResult(
                        label_token=label_token,
                        label_str=label_str,
                        imp_min_coeff=imp_min_coeff,
                        ce_loss_coeff=ce_loss_coeff,
                        steps=steps,
                        label_prob=result.stats.label_prob,
                        l0_total=result.stats.l0_total,
                        l0_per_layer=result.stats.l0_per_layer,
                    ),
                )
                complete_data = {"type": "complete", "data": response_data.model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/cached/{prompt_id}")
@log_errors
def get_cached_graphs(
    prompt_id: int,
    loaded: DepLoadedRun,
    manager: DepStateManager,
) -> list[GraphData | GraphDataWithOptimization]:
    """Get all cached graphs for a prompt.

    Returns list of cached graphs (both standard and optimized) for the given prompt.
    Returns empty list if no cached graphs exist.
    """
    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return []

    token_strings = [loaded.token_strings[t] for t in prompt.token_ids]

    cached_graphs = db.get_all_cached_graphs(prompt_id)

    results: list[GraphData | GraphDataWithOptimization] = []
    for cached, opt_params in cached_graphs:
        edges_typed = [EdgeData(src=e.src, tgt=e.tgt, val=e.val) for e in cached.edges]
        output_probs = {k: OutputProbability(prob=v.prob, token=v.token) for k, v in cached.output_probs.items()}

        if opt_params is None:
            # Standard graph
            results.append(
                GraphData(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
                    cached=True,
                )
            )
        else:
            # Optimized graph
            assert cached.optimization_stats is not None
            results.append(
                GraphDataWithOptimization(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
                    cached=True,
                    optimization=OptimizationResult(
                        label_token=opt_params.label_token,
                        label_str=loaded.token_strings[opt_params.label_token],
                        imp_min_coeff=opt_params.imp_min_coeff,
                        ce_loss_coeff=opt_params.ce_loss_coeff,
                        steps=opt_params.steps,
                        label_prob=cached.optimization_stats.label_prob,
                        l0_total=cached.optimization_stats.l0_total,
                        l0_per_layer=cached.optimization_stats.l0_per_layer,
                    ),
                )
            )

    return results
