import asyncio
import traceback
from contextlib import asynccontextmanager
from functools import wraps

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from spd.app.backend.api import (
    ActivationContext,
    ApplyMaskRequest,
    AvailablePrompt,
    ClusterDashboardResponse,
    CombineMasksRequest,
    CombineMasksResponse,
    ComponentAblationRequest,
    InterventionResponse,
    MaskOverrideDTO,
    Run,
    RunRequest,
    RunResponse,
    SimulateMergeRequest,
    SimulateMergeResponse,
    Status,
    SubcomponentAblationRequest,
    SubcomponentAblationResponse,
    SubcomponentActivationContexts,
    TokenLayerCosineSimilarityData,
)
from spd.app.backend.services.ablation_service import AblationService
from spd.app.backend.services.activation_contexts_service import (
    SubcomponentActivationContextsService,
)
from spd.app.backend.services.cluster_dashboard_service import ComponentActivationContextsService
from spd.app.backend.services.geometry_service import GeometryService
from spd.app.backend.services.run_context_service import (
    CLUSTER_PROJECT,
    ENTITY,
    TRAIN_PROJECT,
    RunContextService,
)

run_context_service = RunContextService()
subcomponent_activations_context_service = SubcomponentActivationContextsService(
    run_context_service
)
component_activation_contexts_service = ComponentActivationContextsService(run_context_service)
ablation_service = AblationService(run_context_service, component_activation_contexts_service)
geometry_service = GeometryService(run_context_service)


def handle_errors(func):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    """Decorator to add error handling with traceback to endpoints.

    Supports both sync and async route handlers.
    """

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e)) from e

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    return sync_wrapper


DEFAULT_RUN_ID = "cztuy3va"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global run_context_service
    # run_context_service.load_run(f"{ENTITY}/{TRAIN_PROJECT}/{DEFAULT_RUN_ID}")
    try:
        yield
    finally:
        pass


app = FastAPI(lifespan=lifespan, debug=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Don't host me publically lol
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/ablate_subcomponents")
@handle_errors
def ablate_subcomponents(request: SubcomponentAblationRequest) -> SubcomponentAblationResponse:
    tokens_logits = ablation_service.ablate_subcomponents(
        request.prompt_id,
        request.subcomponent_mask,
    )
    return SubcomponentAblationResponse(token_logits=tokens_logits)


@app.post("/ablate_components")
@handle_errors
def ablate_components(request: ComponentAblationRequest) -> InterventionResponse:
    tokens_logits, ablation_stats = ablation_service.ablate_components(
        request.prompt_id,
        request.component_mask,
    )
    return InterventionResponse(token_logits=tokens_logits, ablation_stats=ablation_stats)


@app.post("/apply_mask")
@handle_errors
def apply_mask_as_ablation(request: ApplyMaskRequest) -> InterventionResponse:
    """Apply a saved mask as an ablation to a specific prompt."""
    tokens_logits, ablation_stats = ablation_service.run_with_mask_override(
        request.prompt_id, request.mask_override_id
    )
    return InterventionResponse(token_logits=tokens_logits, ablation_stats=ablation_stats)


@app.post("/runs/load/{wandb_run_id}")
@handle_errors
def load_run(wandb_run_id: str):
    global ablation_service
    run_context_service.load_run(f"{ENTITY}/{TRAIN_PROJECT}/{wandb_run_id}")


@app.post("/cluster-runs/load/{wandb_run_id}/{iteration}")
@handle_errors
def load_cluster_run(wandb_run_id: str, iteration: int):
    global ablation_service
    run_context_service.load_cluster_run(f"{ENTITY}/{CLUSTER_PROJECT}/{wandb_run_id}", iteration)


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
    return geometry_service.get_subcomponent_cosine_sims(layer, component_idx)


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
        mask_override=mask_override.to_dto(),
    )


@app.post("/simulate_merge")
@handle_errors
def simulate_merge(request: SimulateMergeRequest) -> SimulateMergeResponse:
    """Simulate merging masks without persisting the result"""
    return ablation_service.get_merge_l0(
        prompt_id=request.prompt_id, layer=request.layer, token_indices=request.token_indices
    )


@app.get("/mask_overrides")
@handle_errors
def get_mask_overrides() -> list[MaskOverrideDTO]:
    return [mo.to_dto() for mo in ablation_service.mask_overrides.values()]


@app.get("/activation_contexts/{layer}/subcomponents")
@handle_errors
async def get_layer_subcomponent_activation_contexts(
    layer: str,
    importance_threshold: float,
    max_examples_per_subcomponent: int,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
) -> list[SubcomponentActivationContexts]:
    f = subcomponent_activations_context_service.get_layer_subcomponents_activation_contexts
    return await f(
        layer=layer,
        importance_threshold=importance_threshold,
        max_examples_per_subcomponent=max_examples_per_subcomponent,
        n_batches=n_batches,
        n_tokens_either_side=n_tokens_either_side,
        batch_size=batch_size,
    )


@app.get("/cluster-dashboard/data")
@handle_errors
async def get_cluster_dashboard_data(
    iteration: int,
    n_samples: int,
    n_batches: int,
    batch_size: int,
    context_length: int,
) -> ClusterDashboardResponse:
    return await component_activation_contexts_service.get_dashboard_data(
        iteration=iteration,
        n_samples=n_samples,
        n_batches=n_batches,
        batch_size=batch_size,
        context_length=context_length,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
