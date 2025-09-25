from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

from spd.app.backend.service import AblationService, LayerCIsDTO, OutputTokenLogitDTO
from spd.models.component_model import ComponentModel, SPDRunInfo

service: AblationService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = AblationService.default()
    try:
        yield
    finally:
        # No explicit teardown required currently
        pass


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
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
    assert service is not None, "Service not initialized"
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


class AblationRequest(BaseModel):
    prompt: str
    component_mask: dict[str, list[list[int]]]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogitDTO]]


@app.post("/modify")
def modify_components(request: AblationRequest) -> AblationResponse:
    assert service is not None, "Service not initialized"
    tokens_logits = service.modify_components(request.prompt, request.component_mask)
    return AblationResponse(token_logits=tokens_logits)


class LoadRequest(BaseModel):
    wandb_run_id: str


@app.post("/load")
def load_run(request: LoadRequest):
    global service

    path = f"wandb:goodfire/spd/runs/{request.wandb_run_id}"
    run_info = SPDRunInfo.from_path(path)

    service = AblationService(
        config=run_info.config,
        cm=ComponentModel.from_run_info(run_info),
        tokenizer=AutoTokenizer.from_pretrained(run_info.config.tokenizer_name),  # pyright: ignore[reportArgumentType]
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
