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


def _get_ci_lookup(prompt_id: int, manager: DepStateManager) -> dict[str, float]:
    """Get CI values for all components from database.

    Args:
        prompt_id: Prompt ID to look up CI values for
        manager: State manager for database access

    Returns:
        Dict mapping component_key to max_ci
    """
    db = manager.db
    conn = db._get_conn()

    rows = conn.execute(
        """SELECT component_key, max_ci
           FROM component_activations
           WHERE prompt_id = ?""",
        (prompt_id,),
    ).fetchall()

    ci_lookup: dict[str, float] = {}
    for row in rows:
        ci_lookup[row["component_key"]] = row["max_ci"]

    return ci_lookup


def filter_edges_by_ci_threshold(
    edges: list[Edge],
    prompt_id: int,
    ci_threshold: float,
    manager: DepStateManager,
) -> list[Edge]:
    """Filter edges by removing those where source or target has CI < ci_threshold.

    Args:
        edges: List of edges to filter
        prompt_id: Prompt ID to look up CI values for
        ci_threshold: Threshold for filtering
        manager: State manager for database access

    Returns:
        Filtered list of edges
    """
    if ci_threshold <= 0:
        return edges

    ci_lookup = _get_ci_lookup(prompt_id, manager)

    return [
        edge
        for edge in edges
        if ci_lookup.get(f"{edge.source.layer}:{edge.source.component_idx}", 0.0) >= ci_threshold
        and ci_lookup.get(f"{edge.target.layer}:{edge.target.component_idx}", 0.0) >= ci_threshold
    ]


def compute_edge_stats(edges: list[EdgeData]) -> tuple[dict[str, float], float]:
    """Compute node importance and max absolute edge value.

    Returns:
        (node_importance, max_abs_attr) where node_importance is sum of squared edge values per node.
    """
    importance: dict[str, float] = {}
    max_abs_attr = 0.0
    for edge in edges:
        val_sq = edge.val * edge.val
        importance[edge.src] = importance.get(edge.src, 0.0) + val_sq
        importance[edge.tgt] = importance.get(edge.tgt, 0.0) + val_sq
        abs_val = abs(edge.val)
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
    ci_threshold: Annotated[float, Query()],
):
    """Compute attribution graph for a prompt with streaming progress."""
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
            # Always compute with ci_threshold=0 to get all edges
            result = compute_local_attributions(
                model=loaded.model,
                tokens=tokens_tensor,
                sources_by_target=loaded.sources_by_target,
                ci_threshold=0.0,
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

                # Store all edges (unfiltered, unnormalized)
                db.save_graph(
                    prompt_id=prompt_id,
                    graph=StoredGraph(
                        edges=raw_edges,
                        output_probs=raw_output_probs,
                    ),
                )

                # Filter edges by ci_threshold before returning to client
                edges = filter_edges_by_ci_threshold(raw_edges, prompt_id, ci_threshold, manager)

                # Apply normalization for response
                edges = _normalize_edges(raw_edges, normalize)
                if len(edges) > GLOBAL_EDGE_LIMIT:
                    edges.sort(key=lambda e: abs(e.strength), reverse=True)
                    edges = edges[:GLOBAL_EDGE_LIMIT]

                edges_typed = [_edge_to_edge_data(e) for e in edges]
                node_importance, max_abs_attr = compute_edge_stats(edges_typed)

                response_data = GraphData(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
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
    ci_threshold: Annotated[float, Query()],
):
    """Compute optimized attribution graph for a prompt with streaming progress."""
    lr = 1e-2

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
            # Always compute with ci_threshold=0 to get all edges
            result = compute_local_attributions_optimized(
                model=loaded.model,
                tokens=tokens_tensor,
                label_token=label_token,
                sources_by_target=loaded.sources_by_target,
                optim_config=optim_config,
                ci_threshold=0.0,
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

                # Store all edges (unfiltered, unnormalized)
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

                # Filter edges by ci_threshold before returning to client
                edges = filter_edges_by_ci_threshold(raw_edges, prompt_id, ci_threshold, manager)

                # Apply normalization for response
                edges = _remove_non_final_output_nodes(raw_edges, len(token_ids))
                edges = _normalize_edges(edges, normalize)
                if len(edges) > GLOBAL_EDGE_LIMIT:
                    edges.sort(key=lambda e: abs(e.strength), reverse=True)
                    edges = edges[:GLOBAL_EDGE_LIMIT]

                edges_typed = [_edge_to_edge_data(e) for e in edges]
                node_importance, max_abs_attr = compute_edge_stats(edges_typed)

                response_data = GraphDataWithOptimization(
                    id=prompt_id,
                    tokens=token_strings,
                    edges=edges_typed,
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


def _remove_non_final_output_nodes(edges: list[Edge], num_tokens: int) -> list[Edge]:
    """Remove edges that are not final output nodes."""
    return [edge for edge in edges if edge.target.seq_pos == num_tokens - 1]


@router.get("/{prompt_id}")
@log_errors
def get_graphs(
    prompt_id: int,
    normalize: Annotated[NormalizeType, Query()],
    ci_threshold: Annotated[float, Query()],
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

    results: list[GraphData | GraphDataWithOptimization] = []
    for graph in stored_graphs:
        # Filter edges by ci_threshold
        edges = filter_edges_by_ci_threshold(
            edges=graph.edges, prompt_id=prompt_id, ci_threshold=ci_threshold, manager=manager
        )
        # Normalize and convert to API format
        edges = _normalize_edges(graph.edges, normalize)
        if len(edges) > GLOBAL_EDGE_LIMIT:
            edges.sort(key=lambda e: abs(e.strength), reverse=True)
            edges = edges[:GLOBAL_EDGE_LIMIT]
        edges_data = [_edge_to_edge_data(e) for e in edges]

        node_importance, max_abs_attr = compute_edge_stats(edges_data)

        if graph.optimization_params is None:
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
