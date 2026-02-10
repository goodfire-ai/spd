"""Intervention forward pass endpoint."""

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.compute import compute_intervention_forward
from spd.app.backend.dependencies import DepDB, DepLoadedRun, DepStateManager
from spd.app.backend.utils import log_errors
from spd.utils.distributed_utils import get_device

# =============================================================================
# Schemas
# =============================================================================


class InterventionNode(BaseModel):
    """A specific node to activate during intervention."""

    layer: str
    seq_pos: int
    component_idx: int


class InterventionRequest(BaseModel):
    """Request for intervention forward pass."""

    text: str
    nodes: list[InterventionNode]
    top_k: int


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str
    token_id: int
    spd_prob: float
    target_prob: float
    logit: float
    target_logit: float


class InterventionResponse(BaseModel):
    """Response from intervention forward pass."""

    input_tokens: list[str]
    predictions_per_position: list[list[TokenPrediction]]


class RunInterventionRequest(BaseModel):
    """Request to run and save an intervention."""

    graph_id: int
    text: str
    selected_nodes: list[str]  # node keys (layer:seq:cIdx)
    top_k: int = 10


class ForkedInterventionRunSummary(BaseModel):
    """Summary of a forked intervention run with modified tokens."""

    id: int
    token_replacements: list[tuple[int, int]]  # [(seq_pos, new_token_id), ...]
    result: InterventionResponse
    created_at: str


class InterventionRunSummary(BaseModel):
    """Summary of a saved intervention run."""

    id: int
    selected_nodes: list[str]
    result: InterventionResponse
    created_at: str
    forked_runs: list[ForkedInterventionRunSummary]


class ForkInterventionRequest(BaseModel):
    """Request to fork an intervention run with modified tokens."""

    token_replacements: list[tuple[int, int]]  # [(seq_pos, new_token_id), ...]
    top_k: int = 10


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
    token_ids = loaded.tokenizer.encode(text)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    active_nodes = [_parse_node_key(key) for key in selected_nodes]

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
            TokenPrediction(
                token=token,
                token_id=token_id,
                spd_prob=spd_prob,
                target_prob=target_prob,
                logit=logit,
                target_logit=target_logit,
            )
            for token, token_id, spd_prob, logit, target_prob, target_logit in pos_predictions
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
    token_ids = loaded.tokenizer.encode(request.text)
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
            TokenPrediction(
                token=token,
                token_id=token_id,
                spd_prob=spd_prob,
                target_prob=target_prob,
                logit=logit,
                target_logit=target_logit,
            )
            for token, token_id, spd_prob, logit, target_prob, target_logit in pos_predictions
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
        forked_runs=[],
    )


@router.get("/runs/{graph_id}")
@log_errors
def get_intervention_runs(graph_id: int, db: DepDB) -> list[InterventionRunSummary]:
    """Get all intervention runs for a graph, including forked runs."""
    records = db.get_intervention_runs(graph_id)
    results = []
    for r in records:
        # Get forked runs for this intervention run
        forked_records = db.get_forked_intervention_runs(r.id)
        forked_runs = [
            ForkedInterventionRunSummary(
                id=fr.id,
                token_replacements=fr.token_replacements,
                result=InterventionResponse.model_validate_json(fr.result_json),
                created_at=fr.created_at,
            )
            for fr in forked_records
        ]

        results.append(
            InterventionRunSummary(
                id=r.id,
                selected_nodes=r.selected_nodes,
                result=InterventionResponse.model_validate_json(r.result_json),
                created_at=r.created_at,
                forked_runs=forked_runs,
            )
        )
    return results


@router.delete("/runs/{run_id}")
@log_errors
def delete_intervention_run(run_id: int, db: DepDB) -> dict[str, bool]:
    """Delete an intervention run."""
    db.delete_intervention_run(run_id)
    return {"success": True}


@router.post("/runs/{run_id}/fork")
@log_errors
def fork_intervention_run(
    run_id: int,
    request: ForkInterventionRequest,
    loaded: DepLoadedRun,
    manager: DepStateManager,
) -> ForkedInterventionRunSummary:
    """Fork an intervention run with modified tokens.

    Takes the same selected_nodes from the parent run, applies token replacements
    to the original prompt, and runs the intervention forward pass.
    """
    db = manager.db

    # Get the parent intervention run
    parent_run = db.get_intervention_run(run_id)
    if parent_run is None:
        raise HTTPException(status_code=404, detail="Intervention run not found")

    # Get the prompt_id from the graph
    conn = db._get_conn()
    row = conn.execute(
        "SELECT prompt_id FROM graphs WHERE id = ?", (parent_run.graph_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    prompt_id = row["prompt_id"]

    # Get the prompt to get original token_ids
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Apply token replacements to get modified token_ids
    modified_token_ids = list(prompt.token_ids)  # Make a copy
    for seq_pos, new_token_id in request.token_replacements:
        if seq_pos < 0 or seq_pos >= len(modified_token_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid seq_pos {seq_pos} for prompt with {len(modified_token_ids)} tokens",
            )
        modified_token_ids[seq_pos] = new_token_id

    # Decode the modified tokens back to text
    modified_text = loaded.tokenizer.decode(modified_token_ids)

    # Run the intervention forward pass with modified tokens but same selected nodes
    response = _run_intervention_forward(
        text=modified_text,
        selected_nodes=parent_run.selected_nodes,
        top_k=request.top_k,
        loaded=loaded,
    )

    # Save the forked run
    fork_id = db.save_forked_intervention_run(
        intervention_run_id=run_id,
        token_replacements=request.token_replacements,
        result_json=response.model_dump_json(),
    )

    # Get the saved record for created_at
    forked_records = db.get_forked_intervention_runs(run_id)
    saved_fork = next((f for f in forked_records if f.id == fork_id), None)
    assert saved_fork is not None

    return ForkedInterventionRunSummary(
        id=fork_id,
        token_replacements=request.token_replacements,
        result=response,
        created_at=saved_fork.created_at,
    )


@router.delete("/forks/{fork_id}")
@log_errors
def delete_forked_intervention_run(fork_id: int, db: DepDB) -> dict[str, bool]:
    """Delete a forked intervention run."""
    db.delete_forked_intervention_run(fork_id)
    return {"success": True}
