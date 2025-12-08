"""Graph computation endpoints for tokenization and attribution graphs."""

import json
import math
import queue
import threading
from collections.abc import Generator
from itertools import groupby
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from spd.app.backend.compute import (
    Edge,
    LocalAttributionResult,
    OptimizedLocalAttributionResult,
    compute_local_attributions,
    compute_local_attributions_optimized,
)
from spd.app.backend.db.database import (
    OptimizationParams,
    OptimizationStats,
    StoredGraph,
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


def compute_edge_stats(edges: list[Edge]) -> tuple[dict[str, float], float]:
    """Compute node importance and max absolute edge value.

    Returns:
        (node_importance, max_abs_attr) where node_importance is sum of squared edge values per node.
    """
    importance: dict[str, float] = {}
    max_abs_attr = 0.0
    for edge in edges:
        val_sq = edge.strength * edge.strength
        src_key = str(edge.source)
        tgt_key = str(edge.target)
        importance[src_key] = importance.get(src_key, 0.0) + val_sq
        importance[tgt_key] = importance.get(tgt_key, 0.0) + val_sq
        abs_val = abs(edge.strength)
        if abs_val > max_abs_attr:
            max_abs_attr = abs_val
    return importance, max_abs_attr


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
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    token_strings = [loaded.token_strings[t] for t in token_ids]
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

                # Build raw edges and output probs for storage
                raw_edges = result.edges
                raw_output_probs_tensor = result.output_probs[0].cpu()
                raw_output_probs: dict[str, OutputProbability] = {}
                for s in range(raw_output_probs_tensor.shape[0]):
                    for c_idx in range(raw_output_probs_tensor.shape[1]):
                        prob = float(raw_output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        raw_output_probs[key] = OutputProbability(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                # Save graph (raw, unnormalized edges)
                db.save_graph(
                    prompt_id=prompt_id,
                    graph=StoredGraph(
                        edges=raw_edges,
                        output_probs=raw_output_probs,
                    ),
                )

                # Process edges for response
                edges_data, node_importance, max_abs_attr = process_edges_for_response(
                    raw_edges, normalize, num_tokens=len(token_ids), is_optimized=False
                )

                response_data = GraphData(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_data,
                    outputProbs=raw_output_probs,
                    nodeImportance=node_importance,
                    maxAbsAttr=max_abs_attr,
                )
                complete_data = {"type": "complete", "data": response_data.model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


def _edge_to_edge_data(edge: Edge) -> EdgeData:
    """Convert Edge (internal format) to EdgeData (API format)."""
    return EdgeData(
        src=str(edge.source),
        tgt=str(edge.target),
        val=edge.strength,
        is_cross_seq=edge.is_cross_seq,
    )


def _normalize_edges(edges: list[Edge], normalize: NormalizeType) -> list[Edge]:
    """Normalize edges by target node or target layer."""
    if normalize == "none":
        return edges

    def get_group_key(edge: Edge) -> str:
        if normalize == "target":
            return str(edge.target)
        return edge.target.layer

    sorted_edges = sorted(edges, key=get_group_key)
    groups = groupby(sorted_edges, key=get_group_key)

    out_edges = []
    for _, group_edges in groups:
        group_edges = list(group_edges)
        group_strength = math.sqrt(sum(edge.strength**2 for edge in group_edges))
        if group_strength == 0:
            continue
        for edge in group_edges:
            out_edges.append(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    is_cross_seq=edge.is_cross_seq,
                    strength=edge.strength / group_strength,
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
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    label_str = loaded.token_strings[label_token]
    token_strings = [loaded.token_strings[t] for t in token_ids]
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    opt_params = OptimizationParams(
        label_token=label_token,
        imp_min_coeff=imp_min_coeff,
        ce_loss_coeff=ce_loss_coeff,
        steps=steps,
        pnorm=pnorm,
    )

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

                # Build raw edges and output probs for storage
                raw_edges = result.edges
                raw_output_probs_tensor = result.output_probs[0].cpu()
                raw_output_probs: dict[str, OutputProbability] = {}
                for s in range(raw_output_probs_tensor.shape[0]):
                    for c_idx in range(raw_output_probs_tensor.shape[1]):
                        prob = float(raw_output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        raw_output_probs[key] = OutputProbability(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                # Save graph (raw, unnormalized edges)
                db.save_graph(
                    prompt_id=prompt_id,
                    graph=StoredGraph(
                        edges=raw_edges,
                        output_probs=raw_output_probs,
                        optimization_params=opt_params,
                        optimization_stats=OptimizationStats(
                            label_prob=result.stats.label_prob,
                            l0_total=result.stats.l0_total,
                            l0_per_layer=result.stats.l0_per_layer,
                        ),
                    ),
                )

                # Process edges for response
                edges_data, node_importance, max_abs_attr = process_edges_for_response(
                    raw_edges, normalize, num_tokens=len(token_ids), is_optimized=True
                )

                response_data = GraphDataWithOptimization(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_data,
                    outputProbs=raw_output_probs,
                    nodeImportance=node_importance,
                    maxAbsAttr=max_abs_attr,
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


def process_edges_for_response(
    edges: list[Edge],
    normalize: NormalizeType,
    num_tokens: int,
    is_optimized: bool,
    edge_limit: int = GLOBAL_EDGE_LIMIT,
) -> tuple[list[EdgeData], dict[str, float], float]:
    """Single source of truth for edge processing pipeline.

    Applies filtering, normalization, limiting, and computes stats.
    Guarantees identical processing for compute and retrieval paths.

    Args:
        edges: Raw edges from computation or database
        normalize: Normalization type ("none", "target", "layer")
        num_tokens: Number of tokens in the prompt (for filtering)
        is_optimized: Whether this is an optimized graph (applies additional filtering)
        edge_limit: Maximum number of edges to return

    Returns:
        (edges_data, node_importance, max_abs_attr)
    """
    if is_optimized:
        final_seq_pos = num_tokens - 1
        edges = [edge for edge in edges if edge.target.seq_pos == final_seq_pos]
    edges = _normalize_edges(edges, normalize)
    node_importance, max_abs_attr = compute_edge_stats(edges)
    # Clip to edge limit for response
    if len(edges) > edge_limit:
        print(f"[WARNING] Edge limit {edge_limit} exceeded ({len(edges)} edges), truncating")
        edges = sorted(edges, key=lambda e: abs(e.strength), reverse=True)[:edge_limit]
    edges_data = [_edge_to_edge_data(e) for e in edges]
    return edges_data, node_importance, max_abs_attr


@router.get("/{prompt_id}")
@log_errors
def get_graphs(
    prompt_id: int,
    normalize: Annotated[NormalizeType, Query()],
    loaded: DepLoadedRun,
    manager: DepStateManager,
) -> list[GraphData | GraphDataWithOptimization]:
    """Get all stored graphs for a prompt.

    Returns list of graphs (both standard and optimized) for the given prompt.
    Returns empty list if no graphs exist.
    """
    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return []

    token_strings = [loaded.token_strings[t] for t in prompt.token_ids]
    stored_graphs = db.get_graphs(prompt_id)

    num_tokens = len(prompt.token_ids)
    results: list[GraphData | GraphDataWithOptimization] = []
    for graph in stored_graphs:
        is_optimized = graph.optimization_params is not None
        edges_data, node_importance, max_abs_attr = process_edges_for_response(
            graph.edges, normalize, num_tokens, is_optimized
        )

        if not is_optimized:
            # Standard graph
            results.append(
                GraphData(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_data,
                    outputProbs=graph.output_probs,
                    nodeImportance=node_importance,
                    maxAbsAttr=max_abs_attr,
                )
            )
        else:
            # Optimized graph
            assert graph.optimization_params is not None
            assert graph.optimization_stats is not None
            results.append(
                GraphDataWithOptimization(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_data,
                    outputProbs=graph.output_probs,
                    nodeImportance=node_importance,
                    maxAbsAttr=max_abs_attr,
                    optimization=OptimizationResult(
                        label_token=graph.optimization_params.label_token,
                        label_str=loaded.token_strings[graph.optimization_params.label_token],
                        imp_min_coeff=graph.optimization_params.imp_min_coeff,
                        ce_loss_coeff=graph.optimization_params.ce_loss_coeff,
                        steps=graph.optimization_params.steps,
                        label_prob=graph.optimization_stats.label_prob,
                        l0_total=graph.optimization_stats.l0_total,
                        l0_per_layer=graph.optimization_stats.l0_per_layer,
                    ),
                )
            )

    return results
