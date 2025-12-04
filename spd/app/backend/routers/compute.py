"""Compute endpoints for tokenization and attribution graphs."""

import json
import queue
import threading
import time
from collections.abc import Generator
from typing import Annotated, Any

import torch
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse

from spd.app.backend.compute import compute_local_attributions, compute_local_attributions_optimized
from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.lib.edge_normalization import normalize_edges_by_target
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
from spd.log import logger
from spd.utils.distributed_utils import get_device

router = APIRouter(prefix="/api", tags=["compute"])

DEVICE = get_device()
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


@router.post("/compute")
@log_errors
def compute_graph(
    token_ids: Annotated[list[int], Body(embed=True)],
    normalize: Annotated[bool, Query()],
    loaded: DepLoadedRun,
):
    """Compute attribution graph for given token IDs."""
    ci_threshold = 1e-6
    output_prob_threshold = 0.01

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    token_strings = [loaded.token_strings[t] for t in token_ids]

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    t_start = time.time()
    result = compute_local_attributions(
        model=loaded.model,
        tokens=tokens_tensor,
        sources_by_target=loaded.sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=loaded.config.sampling,
        device=DEVICE,
        show_progress=False,
    )
    t_end = time.time()

    logger.info(f"[API] /api/compute completed in {t_end - t_start:.2f}s, {len(result.edges)} edges")

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:GLOBAL_EDGE_LIMIT]

    if normalize:
        edges = normalize_edges_by_target(edges)

    edges_typed = [
        EdgeData(src=f"{e[0]}:{e[4]}:{e[2]}", tgt=f"{e[1]}:{e[5]}:{e[3]}", val=e[6]) for e in edges
    ]

    output_probs: dict[str, OutputProbability] = {}
    output_probs_tensor = result.output_probs[0].cpu()

    # Only send output probs above threshold to avoid sending entire vocab
    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            if prob < output_prob_threshold:
                continue
            key = f"{s}:{c_idx}"
            output_probs[key] = OutputProbability(
                prob=round(prob, 6),
                token=loaded.token_strings[c_idx],
            )

    return GraphData(
        id=-1,  # Custom prompts have no ID
        tokens=token_strings,
        edges=edges_typed,
        outputProbs=output_probs,
    )


@router.post("/compute/optimized/stream")
@log_errors
def compute_graph_optimized_stream(
    token_ids: Annotated[list[int], Body(embed=True)],
    label_token: Annotated[int, Query()],
    imp_min_coeff: Annotated[float, Query(gt=0)],
    ce_loss_coeff: Annotated[float, Query(gt=0)],
    steps: Annotated[int, Query(gt=0)],
    pnorm: Annotated[float, Query(gt=0, le=1)],
    normalize: Annotated[bool, Query()],
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)],
    loaded: DepLoadedRun,
):
    """Compute optimized attribution graph for given token IDs with streaming progress."""
    lr = 1e-2
    ci_threshold = 1e-6

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    label_str = loaded.token_strings[label_token]
    token_strings = [loaded.token_strings[t] for t in token_ids]

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
                result = msg["result"]

                edges = result.edges
                edges.sort(key=lambda x: abs(x[6]), reverse=True)
                edges = edges[:GLOBAL_EDGE_LIMIT]

                if normalize:
                    edges = normalize_edges_by_target(edges)

                edges_typed = [
                    EdgeData(src=f"{e[0]}:{e[4]}:{e[2]}", tgt=f"{e[1]}:{e[5]}:{e[3]}", val=e[6])
                    for e in edges
                ]

                output_probs: dict[str, OutputProbability] = {}
                output_probs_tensor = result.output_probs[0].cpu()

                for s in range(output_probs_tensor.shape[0]):
                    for c_idx in range(output_probs_tensor.shape[1]):
                        prob = float(output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        output_probs[key] = OutputProbability(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                response_data = GraphDataWithOptimization(
                    id=-1,  # Custom prompts have no ID
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
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
