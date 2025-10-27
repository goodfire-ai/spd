import traceback
from functools import wraps
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from spd.app.backend.lib.activation_contexts import get_subcomponents_activation_contexts
from spd.app.backend.schemas import ModelActivationContexts, Status
from spd.app.backend.services.run_context_service import RunContextService

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Don't host me publicly lol
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
@handle_errors
def get_subcomponent_activation_contexts(
    importance_threshold: float,
    n_batches: int,
    batch_size: int,
    n_tokens_either_side: int,
    topk_examples: int,
) -> ModelActivationContexts:
    assert (run_context := run_context_service.train_run_context) is not None
    return get_subcomponents_activation_contexts(
        run_context,
        importance_threshold=importance_threshold,
        n_batches=n_batches,
        batch_size=batch_size,
        n_tokens_either_side=n_tokens_either_side,
        topk_examples=topk_examples,
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
