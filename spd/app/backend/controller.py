# %%
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spd.app.backend.service import (
    AblationService,
    LayerCIsDTO,
    OutputTokenLogitDTO,
    StatusDTO,
    TokenLayerCosineSimilarityDataDTO,
)

service = AblationService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    print("🟡 loading initial service")
    service.load_run_from_wandb_id("ry05f67a")
    print("✅ intiial service loaded")
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
    try:
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
    except Exception as e:
        import traceback

        traceback.print_exc()
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e)) from e


class AblationRequest(BaseModel):
    component_mask: dict[str, list[list[int]]]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogitDTO]]


@app.post("/ablate")
def ablate_components(request: AblationRequest) -> AblationResponse:
    try:
        tokens_logits = service.ablate_components(
            request.component_mask,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e)) from e
    return AblationResponse(token_logits=tokens_logits)


class LoadRequest(BaseModel):
    wandb_run_id: str


@app.post("/load")
def load_run(request: LoadRequest):
    global service

    print("🟡 loading service")
    service.load_run_from_wandb_id(request.wandb_run_id)
    print("✅ service loaded")

@app.get("/status")
def get_status() -> StatusDTO:
    return service.get_status()



# TS:
# export type CosineSimilarityData = {
#     input_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
#     output_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
#     component_indices: number[]; // indices corresponding to rows/cols
# };

@app.get("/cosine_similarities")
def get_cosine_similarities(layer: str, token_idx: int) -> TokenLayerCosineSimilarityDataDTO:
    try:
        return service.get_cosine_similarities(layer, token_idx)
    except Exception as e:
        import traceback

        traceback.print_exc()
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e)) from e
# %%

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# # %%

# DEMO_RUN = "wandb:goodfire/spd/runs/ry05f67a"
# service.load_run_from_wandb_id(DEMO_RUN)

# # %%
# # def simple_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> float: # from-scratch kl divergence, assuming both inputs are unnormalized logits


# (
#     _,
#     _,
#     full_run_token_logits, ci_masked_token_logits,
# ) = service.run_prompt(service.get_random_prompt())


# for target, ci_masked in zip(full_run_token_logits[:5], ci_masked_token_logits[:5]):
#     print("-" * 10)
#     for t, c in zip(target, ci_masked):
#         print(f"{t.token}: {t.probability:.4f} | {c.token}: {c.probability:.4f}")
# print(ci_masked_token_logits)
# # %%
