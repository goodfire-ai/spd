# %%
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spd.app.backend.service import (
    AblationService,
    LayerCIsDTO,
    MaskOverrideDTO,
    OutputTokenLogitDTO,
    SparseVectorDTO,
    StatusDTO,
    TokenLayerCosineSimilarityDataDTO,
)

service = AblationService()


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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str


class RunResponse(BaseModel):
    prompt_tokens: list[str]
    layer_cis: list[LayerCIsDTO]
    full_run_token_logits: list[list[OutputTokenLogitDTO]]
    ci_masked_token_logits: list[list[OutputTokenLogitDTO]]


@app.post("/run")
def run_prompt(request: RunRequest) -> RunResponse:
    (
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = service.run_prompt(request.prompt)

    return RunResponse(
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


@app.post("/run_random")
def run_random_prompt() -> RunResponse:
    prompt = service.get_random_prompt()

    (
        prompt_tokens,
        layer_causal_importances,
        full_run_token_logits,
        ci_masked_token_logits,
    ) = service.run_prompt(prompt)

    return RunResponse(
        prompt_tokens=prompt_tokens,
        layer_cis=layer_causal_importances,
        full_run_token_logits=full_run_token_logits,
        ci_masked_token_logits=ci_masked_token_logits,
    )


class AblationRequest(BaseModel):
    component_mask: dict[str, list[list[int]]]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogitDTO]]


@app.post("/ablate")
def ablate_components(request: AblationRequest) -> AblationResponse:
    tokens_logits = service.ablate_components(
        request.component_mask,
    )
    return AblationResponse(token_logits=tokens_logits)


class LoadRequest(BaseModel):
    wandb_run_id: str


@app.post("/load")
def load_run(request: LoadRequest):
    global service
    service.load_run_from_wandb_id(request.wandb_run_id)


@app.get("/status")
def get_status() -> StatusDTO:
    return service.get_status()


@app.get("/cosine_similarities")
def get_cosine_similarities(layer: str, token_idx: int) -> TokenLayerCosineSimilarityDataDTO:
    return service.get_cosine_similarities(layer, token_idx)


class CombineMasksRequest(BaseModel):
    layer: str
    token_indices: list[int]  # List of token indices (positions) to combine
    description: str | None = None


class CombineMasksResponse(BaseModel):
    mask_id: str
    mask_override: MaskOverrideDTO


@app.post("/combine_masks")
def combine_masks(request: CombineMasksRequest) -> CombineMasksResponse:
    mask_override = service.create_combined_mask(
        layer=request.layer, token_indices=request.token_indices, description=request.description
    )

    return CombineMasksResponse(
        mask_id=mask_override.id,
        mask_override=mask_override.to_dto(),
    )


class SimulateMergeRequest(BaseModel):
    layer: str
    token_indices: list[int]


class SimulateMergeResponse(BaseModel):
    l0: int
    jacc: float


@app.post("/simulate_merge")
def simulate_merge(request: SimulateMergeRequest) -> SimulateMergeResponse:
    """Simulate merging masks without persisting the result"""
    l0, jacc = service.get_merge_l0(layer=request.layer, token_indices=request.token_indices)
    return SimulateMergeResponse(l0=l0, jacc=jacc)


@app.get("/mask_overrides")
def get_mask_overrides() -> list[MaskOverrideDTO]:
    return [mo.to_dto() for mo in service.mask_overrides.values()]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
