# %%
import traceback
from contextlib import asynccontextmanager
from functools import wraps

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spd.app.backend.service import (
    AblationService,
    AvailablePromptDTO,
    LayerCIsDTO,
    MaskOverrideDTO,
    OutputTokenLogitDTO,
    StatusDTO,
    TokenLayerCosineSimilarityDataDTO,
)

service = AblationService()


def handle_errors(func):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    """Decorator to add error handling with traceback to endpoints."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    return wrapper


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service.load_run_from_wandb_id("ry05f67a")
    try:
        yield
    finally:
        pass


app = FastAPI(lifespan=lifespan, debug=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str


class RunResponse(BaseModel):
    prompt_id: str
    prompt_tokens: list[str]
    layer_cis: list[LayerCIsDTO]
    full_run_token_logits: list[list[OutputTokenLogitDTO]]
    ci_masked_token_logits: list[list[OutputTokenLogitDTO]]


@app.post("/run")
@handle_errors
def run_prompt(request: RunRequest) -> RunResponse:
    (
        prompt_id,
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = service.run_prompt(request.prompt)

    return RunResponse(
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


@app.get("/available_prompts")
@handle_errors
def get_available_prompts() -> list[AvailablePromptDTO]:
    return service.get_available_prompts()


class RunPromptByIndexRequest(BaseModel):
    dataset_index: int


@app.post("/run_prompt_by_index")
@handle_errors
def run_prompt_by_index(request: RunPromptByIndexRequest) -> RunResponse:
    """Run a specific prompt from the dataset by index."""
    (
        prompt_id,
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = service.run_prompt_by_index(request.dataset_index)

    return RunResponse(
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


class RunWithMaskRequest(BaseModel):
    prompt: str
    mask_override_id: str


@app.post("/run_with_mask")
@handle_errors
def run_prompt_with_mask(request: RunWithMaskRequest) -> RunResponse:
    (
        prompt_id,
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = service.run_prompt_with_mask_override(request.prompt, request.mask_override_id)

    return RunResponse(
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


class RunRandomWithMaskRequest(BaseModel):
    mask_override_id: str


class AblationRequest(BaseModel):
    prompt_id: str
    component_mask: dict[str, list[list[int]]]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogitDTO]]


class ApplyMaskRequest(BaseModel):
    prompt_id: str
    mask_override_id: str


@app.post("/ablate")
@handle_errors
def ablate_components(request: AblationRequest) -> AblationResponse:
    tokens_logits = service.ablate_components(
        request.prompt_id,
        request.component_mask,
    )
    return AblationResponse(token_logits=tokens_logits)


@app.post("/apply_mask")
@handle_errors
def apply_mask_as_ablation(request: ApplyMaskRequest) -> AblationResponse:
    """Apply a saved mask as an ablation to a specific prompt."""
    tokens_logits = service.ablate_with_mask_override(request.prompt_id, request.mask_override_id)
    return AblationResponse(token_logits=tokens_logits)


class LoadRequest(BaseModel):
    wandb_run_id: str


@app.post("/load")
@handle_errors
def load_run(request: LoadRequest):
    global service
    service.load_run_from_wandb_id(request.wandb_run_id)


@app.get("/status")
@handle_errors
def get_status() -> StatusDTO:
    return service.get_status()


@app.get("/cosine_similarities")
@handle_errors
def get_cosine_similarities(
    prompt_id: str, layer: str, token_idx: int
) -> TokenLayerCosineSimilarityDataDTO:
    return service.get_cosine_similarities(prompt_id, layer, token_idx)


class CombineMasksRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]  # List of token indices (positions) to combine
    description: str | None = None


class CombineMasksResponse(BaseModel):
    mask_id: str
    mask_override: MaskOverrideDTO


@app.post("/combine_masks")
@handle_errors
def combine_masks(request: CombineMasksRequest) -> CombineMasksResponse:
    mask_override = service.create_combined_mask(
        prompt_id=request.prompt_id,
        layer=request.layer,
        token_indices=request.token_indices,
        description=request.description,
    )

    return CombineMasksResponse(
        mask_id=mask_override.id,
        mask_override=mask_override.to_dto(),
    )


class SimulateMergeRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]


class SimulateMergeResponse(BaseModel):
    l0: int
    jacc: float


@app.post("/simulate_merge")
@handle_errors
def simulate_merge(request: SimulateMergeRequest) -> SimulateMergeResponse:
    """Simulate merging masks without persisting the result"""
    l0, jacc = service.get_merge_l0(
        prompt_id=request.prompt_id, layer=request.layer, token_indices=request.token_indices
    )
    return SimulateMergeResponse(l0=l0, jacc=jacc)


@app.get("/mask_overrides")
@handle_errors
def get_mask_overrides() -> list[MaskOverrideDTO]:
    return [mo.to_dto() for mo in service.mask_overrides.values()]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
