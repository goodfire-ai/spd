"""Graph computation endpoints for tokenization and attribution graphs."""

import json
import math
import queue
import sys
import threading
import traceback
from collections.abc import Callable, Generator
from itertools import groupby
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from spd.app.backend.compute import (
    Edge,
    compute_prompt_attributions,
    compute_prompt_attributions_optimized,
)
from spd.app.backend.database import GraphType, OptimizationParams, StoredGraph
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.optim_cis import MaskType, OptimCELossConfig, OptimCIConfig, OptimKLLossConfig
from spd.app.backend.schemas import OutputProbability
from spd.app.backend.utils import log_errors
from spd.configs import ImportanceMinimalityLossConfig
from spd.utils.distributed_utils import get_device


class EdgeData(BaseModel):
    """Edge in the attribution graph."""

    src: str  # "layer:seq:cIdx"
    tgt: str  # "layer:seq:cIdx"
    val: float
    is_cross_seq: bool = False


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str


class GraphData(BaseModel):
    """Full attribution graph data."""

    id: int
    graphType: GraphType
    tokens: list[str]
    edges: list[EdgeData]
    outputProbs: dict[str, OutputProbability]
    nodeCiVals: dict[
        str, float
    ]  # node key -> CI value (or output prob for output nodes or 1 for wte node)
    nodeSubcompActs: dict[str, float]  # node key -> subcomponent activation (v_i^T @ a)
    maxAbsAttr: float  # max absolute edge value
    maxAbsSubcompAct: float  # max absolute subcomponent activation for normalization
    l0_total: int  # total active components at current CI threshold


class OptimizationResult(BaseModel):
    """Results from optimized CI computation."""

    imp_min_coeff: float
    steps: int
    pnorm: float
    beta: float
    mask_type: MaskType
    # CE loss params (optional - required together)
    label_token: int | None = None
    label_str: str | None = None
    ce_loss_coeff: float | None = None
    label_prob: float | None = None
    # KL loss param (optional)
    kl_loss_coeff: float | None = None


class GraphDataWithOptimization(GraphData):
    """Attribution graph data with optimization results."""

    optimization: OptimizationResult


class ComponentStats(BaseModel):
    """Statistics for a component across prompts."""

    prompt_count: int
    avg_max_ci: float
    prompt_ids: list[int]


class PromptSearchQuery(BaseModel):
    """Query parameters for prompt search."""

    components: list[str]
    mode: str


class PromptSearchResponse(BaseModel):
    """Response from prompt search endpoint."""

    query: PromptSearchQuery
    count: int
    results: list[PromptPreview]


class TokenizeResponse(BaseModel):
    """Response from tokenize endpoint."""

    token_ids: list[int]
    tokens: list[str]
    text: str


class TokenInfo(BaseModel):
    """A single token from the tokenizer vocabulary."""

    id: int
    string: str


class TokensResponse(BaseModel):
    """Response containing all tokens in the vocabulary."""

    tokens: list[TokenInfo]


# SSE streaming message types
class ProgressMessage(BaseModel):
    """Progress update during streaming computation."""

    type: Literal["progress"]
    current: int
    total: int
    stage: str


class ErrorMessage(BaseModel):
    """Error message during streaming computation."""

    type: Literal["error"]
    error: str


class CompleteMessage(BaseModel):
    """Completion message with result data."""

    type: Literal["complete"]
    data: GraphData


class CompleteMessageWithOptimization(BaseModel):
    """Completion message with optimization result data."""

    type: Literal["complete"]
    data: GraphDataWithOptimization


router = APIRouter(prefix="/api/graphs", tags=["graphs"])

DEVICE = get_device()

# This is a bit of a hack. We want to limit the number of edges returned to avoid overwhelming the frontend.
GLOBAL_EDGE_LIMIT = 5_000


ProgressCallback = Callable[[int, int, str], None]


def build_out_probs(
    ci_masked_out_probs: torch.Tensor,
    ci_masked_out_logits: torch.Tensor,
    target_out_probs: torch.Tensor,
    target_out_logits: torch.Tensor,
    output_prob_threshold: float,
    token_strings: dict[int, str],
) -> dict[str, OutputProbability]:
    """Build output probs dict from CI-masked and target model tensors.

    Filters by CI-masked probability threshold, but includes both probabilities.

    Args:
        ci_masked_out_probs: Shape [seq, vocab] - CI-masked model output probabilities
        ci_masked_out_logits: Shape [seq, vocab] - CI-masked model output logits
        target_out_probs: Shape [seq, vocab] - Target model output probabilities
        target_out_logits: Shape [seq, vocab] - Target model output logits
        output_prob_threshold: Threshold for filtering output probabilities
        token_strings: Dictionary mapping token IDs to strings
    """
    assert ci_masked_out_probs.ndim == 2, f"Expected [seq, vocab], got {ci_masked_out_probs.shape}"
    assert target_out_probs.ndim == 2, f"Expected [seq, vocab], got {target_out_probs.shape}"
    assert ci_masked_out_probs.shape == target_out_probs.shape, (
        f"Shape mismatch: {ci_masked_out_probs.shape} vs {target_out_probs.shape}"
    )

    out_probs: dict[str, OutputProbability] = {}
    for s in range(ci_masked_out_probs.shape[0]):
        for c_idx in range(ci_masked_out_probs.shape[1]):
            prob = float(ci_masked_out_probs[s, c_idx].item())
            if prob < output_prob_threshold:
                continue
            logit = float(ci_masked_out_logits[s, c_idx].item())
            target_prob = float(target_out_probs[s, c_idx].item())
            target_logit = float(target_out_logits[s, c_idx].item())
            key = f"{s}:{c_idx}"
            out_probs[key] = OutputProbability(
                prob=round(prob, 6),
                logit=round(logit, 4),
                target_prob=round(target_prob, 6),
                target_logit=round(target_logit, 4),
                token=token_strings[c_idx],
            )
    return out_probs


def stream_computation(
    work: Callable[[ProgressCallback], GraphData | GraphDataWithOptimization],
) -> StreamingResponse:
    """Run graph computation in a thread with SSE streaming for progress updates."""
    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"type": "progress", "current": current, "total": total, "stage": stage})

    def compute_thread() -> None:
        try:
            result = work(on_progress)
            progress_queue.put({"type": "result", "result": result})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
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
                complete_data = {"type": "complete", "data": msg["result"].model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


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


@router.get("/tokens")
@log_errors
def get_all_tokens(loaded: DepLoadedRun) -> TokensResponse:
    """Get all tokens in the tokenizer vocabulary for client-side search."""
    return TokensResponse(
        tokens=[TokenInfo(id=tid, string=tstr) for tid, tstr in loaded.token_strings.items()]
    )


NormalizeType = Literal["none", "target", "layer"]


def compute_max_abs_attr(edges: list[Edge]) -> float:
    """Compute max absolute edge strength for normalization."""
    if not edges:
        return 0.0
    return max(abs(edge.strength) for edge in edges)


def compute_max_abs_subcomp_act(node_subcomp_acts: dict[str, float]) -> float:
    """Compute max absolute subcomponent activation for normalization."""
    if not node_subcomp_acts:
        return 0.0
    return max(abs(v) for v in node_subcomp_acts.values())


@router.post("")
@log_errors
def compute_graph_stream(
    prompt_id: Annotated[int, Query()],
    normalize: Annotated[NormalizeType, Query()],
    loaded: DepLoadedRun,
    manager: DepStateManager,
    ci_threshold: Annotated[float, Query()],
    included_nodes: Annotated[str | None, Query()] = None,
):
    """Compute attribution graph for a prompt with streaming progress.

    If included_nodes is provided (JSON array of node keys), creates a "manual" graph
    with only those nodes. Otherwise creates a "standard" graph.

    Args:
        included_nodes: JSON array of node keys to include (creates manual graph if provided)
    """
    output_prob_threshold = 0.01

    # Parse and validate included_nodes if provided
    included_nodes_set: set[str] | None = None
    included_nodes_list: list[str] | None = None
    if included_nodes is not None:
        try:
            parsed_nodes = json.loads(included_nodes)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid included_nodes JSON") from e

        if not isinstance(parsed_nodes, list):
            raise HTTPException(status_code=400, detail="included_nodes must be a JSON array")

        if len(parsed_nodes) > 10000:
            raise HTTPException(status_code=400, detail="Too many nodes (max 10000)")

        for node in parsed_nodes:
            if not isinstance(node, str):
                raise HTTPException(status_code=400, detail="All node keys must be strings")
            if len(node) > 100:  # Node keys follow format "layer:seq:cIdx", 100 chars is generous
                raise HTTPException(status_code=400, detail=f"Node key too long: {node[:50]}...")

        included_nodes_list = parsed_nodes
        included_nodes_set = set(parsed_nodes)

    is_manual = included_nodes_set is not None
    graph_type: GraphType = "manual" if is_manual else "standard"

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    token_strings = [loaded.token_strings[t] for t in token_ids]
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    def work(on_progress: ProgressCallback) -> GraphData:
        result = compute_prompt_attributions(
            model=loaded.model,
            tokens=tokens_tensor,
            sources_by_target=loaded.sources_by_target,
            output_prob_threshold=output_prob_threshold,
            sampling=loaded.config.sampling,
            device=DEVICE,
            show_progress=False,
            on_progress=on_progress,
            included_nodes=included_nodes_set,
        )

        out_probs = build_out_probs(
            ci_masked_out_probs=result.ci_masked_out_probs.cpu(),
            ci_masked_out_logits=result.ci_masked_out_logits.cpu(),
            target_out_probs=result.target_out_probs.cpu(),
            target_out_logits=result.target_out_logits.cpu(),
            output_prob_threshold=output_prob_threshold,
            token_strings=loaded.token_strings,
        )
        graph_id = db.save_graph(
            prompt_id=prompt_id,
            graph=StoredGraph(
                graph_type=graph_type,
                edges=result.edges,
                out_probs=out_probs,
                node_ci_vals=result.node_ci_vals,
                node_subcomp_acts=result.node_subcomp_acts,
                included_nodes=included_nodes_list,
            ),
        )

        filtered_node_ci_vals = {k: v for k, v in result.node_ci_vals.items() if v > ci_threshold}
        node_ci_vals_with_pseudo = _add_pseudo_layer_nodes(
            filtered_node_ci_vals, len(token_ids), out_probs
        )
        edges_data, max_abs_attr = process_edges_for_response(
            raw_edges=result.edges,
            normalize=normalize,
            num_tokens=len(token_ids),
            node_ci_vals_with_pseudo=node_ci_vals_with_pseudo,
            is_optimized=False,
        )

        return GraphData(
            id=graph_id,
            graphType=graph_type,
            tokens=token_strings,
            edges=edges_data,
            outputProbs=out_probs,
            nodeCiVals=node_ci_vals_with_pseudo,
            nodeSubcompActs=result.node_subcomp_acts,
            maxAbsAttr=max_abs_attr,
            maxAbsSubcompAct=compute_max_abs_subcomp_act(result.node_subcomp_acts),
            l0_total=len(filtered_node_ci_vals),
        )

    return stream_computation(work)


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
    imp_min_coeff: Annotated[float, Query(gte=0)],
    steps: Annotated[int, Query(gt=0)],
    pnorm: Annotated[float, Query(gt=0)],
    beta: Annotated[float, Query(ge=0)],
    normalize: Annotated[NormalizeType, Query()],
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)],
    loaded: DepLoadedRun,
    manager: DepStateManager,
    ci_threshold: Annotated[float, Query()],
    mask_type: Annotated[MaskType, Query()] = "stochastic",
    # Optional CE loss params (required together)
    label_token: Annotated[int | None, Query()] = None,
    ce_loss_coeff: Annotated[float | None, Query(gt=0)] = None,
    # Optional KL loss param
    kl_loss_coeff: Annotated[float | None, Query(gt=0)] = None,
):
    """Compute optimized attribution graph for a prompt with streaming progress.

    At least one of (ce_loss_coeff, kl_loss_coeff) must be provided.
    If ce_loss_coeff is provided, label_token is also required.
    """
    # Validation
    if ce_loss_coeff is None and kl_loss_coeff is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of ce_loss_coeff or kl_loss_coeff must be provided",
        )
    if ce_loss_coeff is not None and label_token is None:
        raise HTTPException(
            status_code=400,
            detail="label_token is required when ce_loss_coeff is provided",
        )

    lr = 1e-2

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    label_str = loaded.token_strings[label_token] if label_token is not None else None
    token_strings = [loaded.token_strings[t] for t in token_ids]
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    opt_params = OptimizationParams(
        imp_min_coeff=imp_min_coeff,
        steps=steps,
        pnorm=pnorm,
        beta=beta,
        mask_type=mask_type,
        label_token=label_token,
        ce_loss_coeff=ce_loss_coeff,
        kl_loss_coeff=kl_loss_coeff,
    )

    ce_loss_config: OptimCELossConfig | None = None
    if ce_loss_coeff is not None:
        assert label_token is not None
        ce_loss_config = OptimCELossConfig(coeff=ce_loss_coeff, label_token=label_token)
    kl_loss_config: OptimKLLossConfig | None = None
    if kl_loss_coeff is not None:
        kl_loss_config = OptimKLLossConfig(coeff=kl_loss_coeff)

    optim_config = OptimCIConfig(
        seed=0,
        lr=lr,
        steps=steps,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, steps // 4),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=imp_min_coeff, pnorm=pnorm, beta=beta),
        ce_loss_config=ce_loss_config,
        kl_loss_config=kl_loss_config,
        sampling=loaded.config.sampling,
        ce_kl_rounding_threshold=0.5,
        mask_type=mask_type,
    )

    def work(on_progress: ProgressCallback) -> GraphDataWithOptimization:
        result = compute_prompt_attributions_optimized(
            model=loaded.model,
            tokens=tokens_tensor,
            sources_by_target=loaded.sources_by_target,
            optim_config=optim_config,
            output_prob_threshold=output_prob_threshold,
            device=DEVICE,
            show_progress=False,
            on_progress=on_progress,
        )

        out_probs = build_out_probs(
            ci_masked_out_probs=result.ci_masked_out_probs.cpu(),
            ci_masked_out_logits=result.ci_masked_out_logits.cpu(),
            target_out_probs=result.target_out_probs.cpu(),
            target_out_logits=result.target_out_logits.cpu(),
            output_prob_threshold=output_prob_threshold,
            token_strings=loaded.token_strings,
        )
        graph_id = db.save_graph(
            prompt_id=prompt_id,
            graph=StoredGraph(
                graph_type="optimized",
                edges=result.edges,
                out_probs=out_probs,
                node_ci_vals=result.node_ci_vals,
                node_subcomp_acts=result.node_subcomp_acts,
                optimization_params=opt_params,
                label_prob=result.label_prob,
            ),
        )

        filtered_node_ci_vals = {k: v for k, v in result.node_ci_vals.items() if v > ci_threshold}
        node_ci_vals_with_pseudo = _add_pseudo_layer_nodes(
            filtered_node_ci_vals, len(token_ids), out_probs
        )
        edges_data, max_abs_attr = process_edges_for_response(
            raw_edges=result.edges,
            normalize=normalize,
            num_tokens=len(token_ids),
            node_ci_vals_with_pseudo=node_ci_vals_with_pseudo,
            is_optimized=True,
        )

        return GraphDataWithOptimization(
            id=graph_id,
            graphType="optimized",
            tokens=token_strings,
            edges=edges_data,
            outputProbs=out_probs,
            nodeCiVals=node_ci_vals_with_pseudo,
            nodeSubcompActs=result.node_subcomp_acts,
            maxAbsAttr=max_abs_attr,
            maxAbsSubcompAct=compute_max_abs_subcomp_act(result.node_subcomp_acts),
            l0_total=len(filtered_node_ci_vals),
            optimization=OptimizationResult(
                imp_min_coeff=imp_min_coeff,
                steps=steps,
                pnorm=pnorm,
                beta=beta,
                mask_type=mask_type,
                label_token=label_token,
                label_str=label_str,
                ce_loss_coeff=ce_loss_coeff,
                label_prob=result.label_prob,
                kl_loss_coeff=kl_loss_coeff,
            ),
        )

    return stream_computation(work)


def _add_pseudo_layer_nodes(
    node_ci_vals: dict[str, float],
    num_tokens: int,
    out_probs: dict[str, OutputProbability],
) -> dict[str, float]:
    """Add wte and output pseudo-nodes for simpler rendering and filtering logic.

    wte nodes get CI=1.0 (always visible), output nodes use their CI-masked probability.
    """
    result = dict(node_ci_vals)
    for seq_pos in range(num_tokens):
        result[f"wte:{seq_pos}:0"] = 1.0
    for key, out_prob in out_probs.items():
        seq_pos, token_id = key.split(":")
        result[f"output:{seq_pos}:{token_id}"] = out_prob.prob
    return result


def process_edges_for_response(
    raw_edges: list[Edge],
    normalize: NormalizeType,
    num_tokens: int,
    node_ci_vals_with_pseudo: dict[str, float],
    is_optimized: bool,
    edge_limit: int = GLOBAL_EDGE_LIMIT,
) -> tuple[list[EdgeData], float]:
    """Process edges: filter by CI, normalize, and limit."""

    # Filter to final seq position for optimized graphs
    if is_optimized:
        final_seq_pos = num_tokens - 1
        raw_edges = [e for e in raw_edges if e.target.seq_pos == final_seq_pos]

    # Only include edges that connect to nodes in node_ci_vals_with_pseudo
    node_keys = set(node_ci_vals_with_pseudo.keys())
    edges = [e for e in raw_edges if str(e.source) in node_keys and str(e.target) in node_keys]

    edges = _normalize_edges(edges=edges, normalize=normalize)
    max_abs_attr = compute_max_abs_attr(edges=edges)

    if len(edges) > edge_limit:
        print(f"[WARNING] Edge limit {edge_limit} exceeded ({len(edges)} edges), truncating")
        edges = sorted(edges, key=lambda e: abs(e.strength), reverse=True)[:edge_limit]

    edges_data = [_edge_to_edge_data(e) for e in edges]

    return edges_data, max_abs_attr


def stored_graph_to_response(
    graph: StoredGraph,
    token_ids: list[int],
    token_strings_map: dict[int, str],
    normalize: NormalizeType,
    ci_threshold: float,
) -> GraphData | GraphDataWithOptimization:
    """Convert a StoredGraph to API response format."""
    token_strings = [token_strings_map[t] for t in token_ids]
    num_tokens = len(token_ids)
    is_optimized = graph.optimization_params is not None

    filtered_node_ci_vals = {k: v for k, v in graph.node_ci_vals.items() if v > ci_threshold}
    l0_total = len(filtered_node_ci_vals)

    node_ci_vals_with_pseudo = _add_pseudo_layer_nodes(
        filtered_node_ci_vals, num_tokens, graph.out_probs
    )
    edges_data, max_abs_attr = process_edges_for_response(
        raw_edges=graph.edges,
        normalize=normalize,
        num_tokens=num_tokens,
        node_ci_vals_with_pseudo=node_ci_vals_with_pseudo,
        is_optimized=is_optimized,
    )

    if not is_optimized:
        return GraphData(
            id=graph.id,
            graphType=graph.graph_type,
            tokens=token_strings,
            edges=edges_data,
            outputProbs=graph.out_probs,
            nodeCiVals=node_ci_vals_with_pseudo,
            nodeSubcompActs=graph.node_subcomp_acts,
            maxAbsAttr=max_abs_attr,
            maxAbsSubcompAct=compute_max_abs_subcomp_act(graph.node_subcomp_acts),
            l0_total=l0_total,
        )

    assert graph.optimization_params is not None

    label_str: str | None = None
    if graph.optimization_params.label_token is not None:
        label_str = token_strings_map[graph.optimization_params.label_token]

    return GraphDataWithOptimization(
        id=graph.id,
        graphType=graph.graph_type,
        tokens=token_strings,
        edges=edges_data,
        outputProbs=graph.out_probs,
        nodeCiVals=node_ci_vals_with_pseudo,
        nodeSubcompActs=graph.node_subcomp_acts,
        maxAbsAttr=max_abs_attr,
        maxAbsSubcompAct=compute_max_abs_subcomp_act(graph.node_subcomp_acts),
        l0_total=l0_total,
        optimization=OptimizationResult(
            imp_min_coeff=graph.optimization_params.imp_min_coeff,
            steps=graph.optimization_params.steps,
            pnorm=graph.optimization_params.pnorm,
            beta=graph.optimization_params.beta,
            mask_type=graph.optimization_params.mask_type,
            label_token=graph.optimization_params.label_token,
            label_str=label_str,
            ce_loss_coeff=graph.optimization_params.ce_loss_coeff,
            label_prob=graph.label_prob,
            kl_loss_coeff=graph.optimization_params.kl_loss_coeff,
        ),
    )


@router.get("/{prompt_id}")
@log_errors
def get_graphs(
    prompt_id: int,
    normalize: Annotated[NormalizeType, Query()],
    ci_threshold: Annotated[float, Query(ge=0)],
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

    stored_graphs = db.get_graphs(prompt_id)
    return [
        stored_graph_to_response(
            graph=graph,
            token_ids=prompt.token_ids,
            token_strings_map=loaded.token_strings,
            normalize=normalize,
            ci_threshold=ci_threshold,
        )
        for graph in stored_graphs
    ]
