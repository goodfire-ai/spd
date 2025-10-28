import json
import traceback
from collections.abc import Generator
from functools import wraps
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.backend.lib.activation_contexts import (
    ActivationsData,
    get_activations_data_streaming,
    map_to_model_activations_contexts,
)
from app.backend.schemas import Status
from app.backend.services.run_context_service import RunContextService

run_context_service = RunContextService()


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

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    # extremely permissive CORS policy for now as we're only running locally
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
@handle_errors
def get_status() -> Status:
    return run_context_service.get_status()


@app.post("/runs/load")
@handle_errors
def load_run(wandb_run_path: str):
    run_context_service.load_run(unquote(wandb_run_path))


@app.get("/activation_contexts/subcomponents")
def get_subcomponent_activation_contexts(
    importance_threshold: float,
    n_batches: int,
    batch_size: int,
    n_tokens_either_side: int,
    topk_examples: int,
) -> StreamingResponse:
    run_context = run_context_service.train_run_context
    if run_context is None:
        raise HTTPException(status_code=400, detail="No training run loaded")

    def generate() -> Generator[str]:
        for res in get_activations_data_streaming(
            run_context,
            importance_threshold,
            n_batches,
            n_tokens_either_side,
            batch_size,
            topk_examples,
        ):
            match res:
                case ("progress", data):
                    assert isinstance(data, int)
                    progress_data = {"type": "progress", "current": data, "total": n_batches}
                    yield f"data: {json.dumps(progress_data)}\n\n"
                case ("complete", data):
                    assert isinstance(data, ActivationsData)
                    result = map_to_model_activations_contexts(run_context.tokenizer, data)
                    complete_data = {"type": "complete", "result": result.model_dump()}
                    yield f"data: {json.dumps(complete_data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
