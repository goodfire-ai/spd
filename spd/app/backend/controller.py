import traceback
from functools import wraps

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from spd.app.backend.api import (
    AblationResponse,
    ApplyMaskRequest,
    AvailablePrompt,
    CombineMasksRequest,
    CombineMasksResponse,
    MaskDTO,
    ModelActivationContexts,
    RunResponse,
    SimulateMergeRequest,
    SimulateMergeResponse,
    Status,
    SubcomponentAblationRequest,
    SubcomponentAblationResponse,
)
from spd.app.backend.lib.activation_contexts import get_subcomponents_activation_contexts
from spd.app.backend.services.ablation_service import AblationService
from spd.app.backend.services.run_context_service import ENTITY, TRAIN_PROJECT, RunContextService

run_context_service = RunContextService()
ablation_service = AblationService(run_context_service)


def handle_errors(func):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    """Decorator to add error handling with traceback to endpoints"""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    return sync_wrapper


app = FastAPI(debug=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Don't host me publicly lol
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
@handle_errors
def get_status() -> Status:
    return run_context_service.get_status()


@app.post("/runs/load/{wandb_run_id}")
@handle_errors
def load_run(wandb_run_id: str):
    global ablation_service
    run_context_service.load_run(f"{ENTITY}/{TRAIN_PROJECT}/{wandb_run_id}")


@app.get("/available_prompts")
@handle_errors
def get_available_prompts() -> list[AvailablePrompt]:
    return run_context_service.get_available_prompts()


@app.post("/run_prompt/{dataset_index}")
@handle_errors
def run_prompt_by_index(dataset_index: int) -> RunResponse:
    """Run a specific prompt from the dataset by index."""
    return ablation_service.run_prompt_by_index(dataset_index)


@app.post("/ablate_subcomponents")
@handle_errors
def ablate_subcomponents(request: SubcomponentAblationRequest) -> SubcomponentAblationResponse:
    tokens_logits = ablation_service.ablate_subcomponents(
        request.prompt_id,
        request.subcomponent_mask,
    )
    return SubcomponentAblationResponse(token_logits=tokens_logits)


@app.post("/simulate_merge")
@handle_errors
def simulate_merge(request: SimulateMergeRequest) -> SimulateMergeResponse:
    """Simulate merging masks without persisting the result"""
    return ablation_service.get_merge_l0(
        prompt_id=request.prompt_id, layer=request.layer, token_indices=request.token_indices
    )


@app.post("/combine_masks")
@handle_errors
def combine_masks(request: CombineMasksRequest) -> CombineMasksResponse:
    mask = ablation_service.create_combined_mask(
        prompt_id=request.prompt_id,
        layer=request.layer,
        token_indices=request.token_indices,
        description=request.description,
    )

    ablation_service.save_mask(mask)

    return CombineMasksResponse(
        mask_id=mask.id,
        mask=mask.to_dto(),
    )


@app.get("/mask")
@handle_errors
def get_masks() -> list[MaskDTO]:
    return [mo.to_dto() for mo in ablation_service.saved_masks.values()]


@app.post("/apply_mask")
@handle_errors
def apply_mask_as_ablation(request: ApplyMaskRequest) -> AblationResponse:
    """Apply a saved mask as an ablation to a specific prompt."""
    tokens_logits, ablation_effect = ablation_service.run_with_mask(
        request.prompt_id, request.mask_id
    )
    return AblationResponse(token_logits=tokens_logits, ablation_effect=ablation_effect)


@app.get("/activation_contexts/subcomponents")
@handle_errors
def get_subcomponent_activation_contexts(
    importance_threshold: float,
    max_examples_per_subcomponent: int,
    n_batches: int,
    batch_size: int,
    n_tokens_either_side: int,
) -> ModelActivationContexts:
    assert (run_context := run_context_service.train_run_context) is not None
    return get_subcomponents_activation_contexts(
        run_context,
        importance_threshold=importance_threshold,
        max_examples_per_subcomponent=max_examples_per_subcomponent,
        n_batches=n_batches,
        n_tokens_either_side=n_tokens_either_side,
        batch_size=batch_size,
    )


@app.get("/")
@handle_errors
def healthcheck() -> str:
    return "Hello, World!"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SPD backend server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
