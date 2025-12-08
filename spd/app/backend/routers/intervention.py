"""Intervention forward pass endpoint."""

import torch
from fastapi import APIRouter

from spd.app.backend.compute import compute_intervention_forward
from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.schemas import (
    InterventionRequest,
    InterventionResponse,
    TokenPrediction,
)
from spd.app.backend.utils import log_errors
from spd.utils.distributed_utils import get_device

router = APIRouter(prefix="/api/intervention", tags=["intervention"])

DEVICE = get_device()


@router.post("")
@log_errors
def run_intervention(request: InterventionRequest, loaded: DepLoadedRun) -> InterventionResponse:
    """Run intervention forward pass with specified nodes active."""
    # Tokenize
    token_ids = loaded.tokenizer.encode(request.text, add_special_tokens=False)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    # Convert request nodes to tuples
    active_nodes = [(n.layer, n.seq_pos, n.component_idx) for n in request.nodes]

    # Validate seq positions are within bounds
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

    # Convert to response schema
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
