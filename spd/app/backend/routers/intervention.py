"""Intervention forward pass endpoint."""

import torch
from fastapi import APIRouter

from spd.app.backend.compute import compute_intervention_forward
from spd.app.backend.dependencies import DepDB, DepLoadedRun
from spd.app.backend.schemas import (
    InterventionRequest,
    InterventionResponse,
    InterventionRunSummary,
    RunInterventionRequest,
    TokenPrediction,
)
from spd.app.backend.utils import log_errors
from spd.utils.distributed_utils import get_device

router = APIRouter(prefix="/api/intervention", tags=["intervention"])

DEVICE = get_device()


def _parse_node_key(key: str) -> tuple[str, int, int]:
    """Parse 'layer:seq:cIdx' into (layer, seq_pos, component_idx)."""
    parts = key.split(":")
    assert len(parts) == 3, f"Invalid node key format: {key!r} (expected 'layer:seq:cIdx')"
    layer, seq_str, cidx_str = parts
    # wte and output are pseudo-layers for visualization only - not interventable
    assert layer not in ("wte", "output"), (
        f"Cannot intervene on {layer!r} nodes - only internal layers (attn/mlp) are interventable"
    )
    return layer, int(seq_str), int(cidx_str)


def _run_intervention_forward(
    text: str,
    selected_nodes: list[str],
    top_k: int,
    loaded: DepLoadedRun,
) -> InterventionResponse:
    """Run intervention forward pass and return response."""
    token_ids = loaded.tokenizer.encode(text, add_special_tokens=False)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    active_nodes = [_parse_node_key(key) for key in selected_nodes]

    for layer, seq_pos, cidx in active_nodes:
        print(f"Intervening on {layer}:{seq_pos}:{cidx}")

    seq_len = tokens.shape[1]
    for _, seq_pos, _ in active_nodes:
        if seq_pos >= seq_len:
            raise ValueError(f"seq_pos {seq_pos} out of bounds for text with {seq_len} tokens")

    result = compute_intervention_forward(
        model=loaded.model,
        tokens=tokens,
        active_nodes=active_nodes,
        top_k=top_k,
        tokenizer=loaded.tokenizer,
    )

    predictions_per_position = [
        [
            TokenPrediction(token=token, token_id=token_id, prob=prob, logit=logit)
            for token, token_id, prob, logit in pos_predictions
        ]
        for pos_predictions in result.predictions_per_position
    ]

    return InterventionResponse(
        input_tokens=result.input_tokens,
        predictions_per_position=predictions_per_position,
    )


@router.post("")
@log_errors
def run_intervention(request: InterventionRequest, loaded: DepLoadedRun) -> InterventionResponse:
    """Run intervention forward pass with specified nodes active (legacy endpoint)."""
    token_ids = loaded.tokenizer.encode(request.text, add_special_tokens=False)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    active_nodes = [(n.layer, n.seq_pos, n.component_idx) for n in request.nodes]

    seq_len = tokens.shape[1]
    for _, seq_pos, _ in active_nodes:
        if seq_pos >= seq_len:
            raise ValueError(f"seq_pos {seq_pos} out of bounds for text with {seq_len} tokens")

    result = compute_intervention_forward(
        model=loaded.model,
        tokens=tokens,
        active_nodes=active_nodes,
        top_k=request.top_k,
        tokenizer=loaded.tokenizer,
    )

    predictions_per_position = [
        [
            TokenPrediction(token=token, token_id=token_id, prob=prob, logit=logit)
            for token, token_id, prob, logit in pos_predictions
        ]
        for pos_predictions in result.predictions_per_position
    ]

    return InterventionResponse(
        input_tokens=result.input_tokens,
        predictions_per_position=predictions_per_position,
    )


@router.post("/run")
@log_errors
def run_and_save_intervention(
    request: RunInterventionRequest,
    loaded: DepLoadedRun,
    db: DepDB,
) -> InterventionRunSummary:
    """Run an intervention and save the result."""
    response = _run_intervention_forward(
        text=request.text,
        selected_nodes=request.selected_nodes,
        top_k=request.top_k,
        loaded=loaded,
    )

    run_id = db.save_intervention_run(
        graph_id=request.graph_id,
        selected_nodes=request.selected_nodes,
        result_json=response.model_dump_json(),
    )

    record = db.get_intervention_runs(request.graph_id)
    saved_run = next((r for r in record if r.id == run_id), None)
    assert saved_run is not None

    return InterventionRunSummary(
        id=run_id,
        selected_nodes=request.selected_nodes,
        result=response,
        created_at=saved_run.created_at,
    )


@router.get("/runs/{graph_id}")
@log_errors
def get_intervention_runs(graph_id: int, db: DepDB) -> list[InterventionRunSummary]:
    """Get all intervention runs for a graph."""
    records = db.get_intervention_runs(graph_id)
    return [
        InterventionRunSummary(
            id=r.id,
            selected_nodes=r.selected_nodes,
            result=InterventionResponse.model_validate_json(r.result_json),
            created_at=r.created_at,
        )
        for r in records
    ]


@router.delete("/runs/{run_id}")
@log_errors
def delete_intervention_run(run_id: int, db: DepDB) -> dict[str, bool]:
    """Delete an intervention run."""
    db.delete_intervention_run(run_id)
    return {"success": True}
