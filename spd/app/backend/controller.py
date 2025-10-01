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
    MaskOverride,
    OutputTokenLogit,
    RunResponse,
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
    Run,
    RunContextService,
    Status,
)

run_context_service = RunContextService()

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
        #
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        #
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str


@app.post("/run")
@handle_errors
def run_prompt(request: RunRequest) -> RunResponse:
    return ablation_service.run_prompt(request.prompt)


@app.get("/available_prompts")
@handle_errors
def get_available_prompts() -> list[AvailablePrompt]:
    return run_context_service.get_available_prompts()


@app.post("/run_prompt/{dataset_index}")
@handle_errors
def run_prompt_by_index(dataset_index: int) -> RunResponse:
    """Run a specific prompt from the dataset by index."""
    return ablation_service.run_prompt_by_index(dataset_index)


class SubcomponentAblationRequest(BaseModel):
    prompt_id: str
    subcomponent_mask: dict[str, list[list[int]]]


class SubcomponentAblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]


@app.post("/ablate_subcomponents")
@handle_errors
def ablate_subcomponents(request: SubcomponentAblationRequest) -> SubcomponentAblationResponse:
    tokens_logits = ablation_service.ablate_subcomponents(
        request.prompt_id,
        request.subcomponent_mask,
    )
    return SubcomponentAblationResponse(token_logits=tokens_logits)


class ComponentAblationRequest(BaseModel):
    prompt_id: str
    component_mask: dict[str, list[list[int]]]


class InterventionResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]


@app.post("/ablate_components")
@handle_errors
def ablate_components(request: ComponentAblationRequest) -> InterventionResponse:
    tokens_logits = ablation_service.ablate_components(
        request.prompt_id,
        request.component_mask,
    )
    return InterventionResponse(token_logits=tokens_logits)


class ApplyMaskRequest(BaseModel):
    prompt_id: str
    mask_override_id: str


@app.post("/apply_mask")
@handle_errors
def apply_mask_as_ablation(request: ApplyMaskRequest) -> InterventionResponse:
    """Apply a saved mask as an ablation to a specific prompt."""
    return InterventionResponse(
        token_logits=ablation_service.run_with_mask_override(
            request.prompt_id, request.mask_override_id
        )
    )


@app.post("/runs/load/{wandb_run_id}")
@handle_errors
def load_run(wandb_run_id: str):
    global ablation_service
    run_context_service.load_run_from_wandb_id(wandb_run_id)


@app.get("/runs")
@handle_errors
def get_wandb_runs() -> list[Run]:
    return run_context_service.get_runs()


@app.get("/status")
@handle_errors
def get_status() -> Status:
    return run_context_service.get_status()


@app.get("/cosine_similarities/{layer}/{component_idx}")
@handle_errors
def get_cosine_similarities(layer: str, component_idx: int) -> TokenLayerCosineSimilarityData:
    return ablation_service.get_subcomponent_cosine_sims(layer, component_idx)


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
    component_example_sets: list[ComponentActivationContexts]


@app.get("/component_activation_contexts/{layer}")
@handle_errors
def get_layer_activation_contexts(
    layer: str,
) -> GetLayerActivationContextsResponse:
    res = GetLayerActivationContextsResponse(
        layer=layer,
        component_example_sets=component_activations_service.get_layer_activation_contexts(layer),
    )
    print(type(res))
    return res


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
