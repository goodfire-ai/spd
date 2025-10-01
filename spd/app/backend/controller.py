# %%
import traceback
from contextlib import asynccontextmanager
from functools import wraps

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spd.app.backend.services.ablation_service import (
    AblationService,
    LayerCIs,
    MaskOverride,
    OutputTokenLogit,
    SparseVector,
    TokenLayerCosineSimilarityData,
)
from spd.app.backend.services.component_activations_service import (
    ActivationContext,
    ComponentActivationContexts,
    ComponentActivationContextsService,
)
from spd.app.backend.services.run_context_service import (
    AvailablePrompt,
    RunContextService,
    Status,
)
from spd.app.backend.services.wandb_service import Run, WandBService

run_context_service = RunContextService()

wandb_service = WandBService()
ablation_service = AblationService(run_context_service)
component_activations_service = ComponentActivationContextsService(run_context_service)


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
    global run_context_service
    run_context_service.load_run_from_wandb_id("ry05f67a")
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
    layer_cis: list[LayerCIs]
    full_run_token_logits: list[list[OutputTokenLogit]]
    ci_masked_token_logits: list[list[OutputTokenLogit]]


@app.post("/run")
@handle_errors
def run_prompt(request: RunRequest) -> RunResponse:
    (
        prompt_id,
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = ablation_service.run_prompt(request.prompt)

    return RunResponse(
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


@app.get("/available_prompts")
@handle_errors
def get_available_prompts() -> list[AvailablePrompt]:
    return run_context_service.get_available_prompts()


@app.post("/run_prompt/{dataset_index}")
@handle_errors
def run_prompt_by_index(dataset_index: int) -> RunResponse:
    """Run a specific prompt from the dataset by index."""
    (
        prompt_id,
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = ablation_service.run_prompt_by_index(dataset_index)

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
    ) = ablation_service.run_prompt_with_mask_override(request.prompt, request.mask_override_id)

    return RunResponse(
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


class AblationRequest(BaseModel):
    prompt_id: str
    component_mask: dict[str, list[list[int]]]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]


@app.post("/ablate")
@handle_errors
def ablate_components(request: AblationRequest) -> AblationResponse:
    tokens_logits = ablation_service.ablate_components(
        request.prompt_id,
        request.component_mask,
    )
    return AblationResponse(token_logits=tokens_logits)


class ApplyMaskRequest(BaseModel):
    prompt_id: str
    mask_override_id: str


@app.post("/apply_mask")
@handle_errors
def apply_mask_as_ablation(request: ApplyMaskRequest) -> AblationResponse:
    """Apply a saved mask as an ablation to a specific prompt."""
    tokens_logits = ablation_service.ablate_with_mask_override(
        request.prompt_id, request.mask_override_id
    )
    return AblationResponse(token_logits=tokens_logits)


@app.post("/load/{wandb_run_id}")
@handle_errors
def load_run(wandb_run_id: str):
    global ablation_service
    run_context_service.load_run_from_wandb_id(wandb_run_id)


@app.get("/status")
@handle_errors
def get_status() -> Status:
    return run_context_service.get_status()


@app.get("/cosine_similarities")
@handle_errors
def get_cosine_similarities(
    prompt_id: str, layer: str, token_idx: int
) -> TokenLayerCosineSimilarityData:
    return ablation_service.get_cosine_similarities_of_active_components(
        prompt_id, layer, token_idx
    )


class CombineMasksRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]  # List of token indices (positions) to combine
    description: str | None = None


class MaskOverrideDTO(BaseModel):
    id: str
    layer: str
    combined_mask: SparseVector
    description: str | None

    @classmethod
    def from_mask_override(cls, mask_override: MaskOverride) -> "MaskOverrideDTO":
        return MaskOverrideDTO(
            id=mask_override.id,
            description=mask_override.description,
            layer=mask_override.layer,
            combined_mask=SparseVector.from_tensor(mask_override.combined_mask),
        )


class CombineMasksResponse(BaseModel):
    mask_id: str
    mask_override: MaskOverrideDTO


@app.post("/combine_masks")
@handle_errors
def combine_masks(request: CombineMasksRequest) -> CombineMasksResponse:
    mask_override = ablation_service.create_combined_mask(
        prompt_id=request.prompt_id,
        layer=request.layer,
        token_indices=request.token_indices,
        description=request.description,
    )

    return CombineMasksResponse(
        mask_id=mask_override.id,
        mask_override=MaskOverrideDTO.from_mask_override(mask_override),
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
    l0, jacc = ablation_service.get_merge_l0(
        prompt_id=request.prompt_id, layer=request.layer, token_indices=request.token_indices
    )
    return SimulateMergeResponse(l0=l0, jacc=jacc)


@app.get("/mask_overrides")
@handle_errors
def get_mask_overrides() -> list[MaskOverrideDTO]:
    return [
        MaskOverrideDTO.from_mask_override(mo) for mo in ablation_service.mask_overrides.values()
    ]


class GetLayerActivationContextsResponse(BaseModel):
    layer: str
    component_examples: list[ComponentActivationContexts]


@app.get("/component_activation_contexts/{layer}")
@handle_errors
def get_layer_activation_contexts(
    layer: str,
) -> GetLayerActivationContextsResponse:
    return GetLayerActivationContextsResponse(
        layer=layer,
        component_examples=component_activations_service.get_layer_activation_contexts(layer),
    )


class GetComponentActivationContextsResponse(BaseModel):
    layer: str
    component_idx: int
    examples: list[ActivationContext]


@app.get("/component_activation_contexts/{layer}/{component_idx}")
@handle_errors
def get_component_activation_contexts(
    layer: str,
    component_idx: int,
) -> GetComponentActivationContextsResponse:
    contexts = component_activations_service.get_component_activation_contexts(
        component_idx=component_idx,
        layer=layer,
    )
    return GetComponentActivationContextsResponse(
        layer=layer,
        component_idx=component_idx,
        examples=contexts,
    )


@app.get("/wandb_runs")
@handle_errors
def get_wandb_runs() -> list[Run]:
    return wandb_service.get_runs()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
