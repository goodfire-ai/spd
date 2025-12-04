"""Unified FastAPI server for the SPD app.

Merges the main app backend with the local attributions server.
Supports multiple runs, on-demand attribution graph computation,
and activation contexts generation.

Usage:
    python -m spd.app.backend.server --port 8000
"""

from contextlib import asynccontextmanager

import fire
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from spd.app.backend.db import LocalAttrDB
from spd.app.backend.routers import (
    activation_contexts_router,
    compute_router,
    graphs_router,
    runs_router,
)
from spd.app.backend.state import StateManager
from spd.log import logger
from spd.utils.distributed_utils import get_device

DEVICE = get_device()


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    """Initialize DB connection at startup. Model loaded on-demand via /api/runs/load."""
    manager = StateManager.get()

    db = LocalAttrDB(check_same_thread=False)
    db.init_schema()
    manager.initialize(db)

    logger.info(f"[STARTUP] DB initialized: {db.db_path}")
    logger.info(f"[STARTUP] Device: {DEVICE}")
    logger.info(f"[STARTUP] CUDA available: {torch.cuda.is_available()}")

    runs = db.get_all_runs()
    logger.info(f"[STARTUP] Found {len(runs)} runs in database")
    for run in runs:
        logger.info(f"  - Run {run.id}: {run.wandb_path}")

    yield

    manager.close()


app = FastAPI(title="SPD App API", lifespan=lifespan, debug=True)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routers
app.include_router(runs_router)
app.include_router(graphs_router)
app.include_router(compute_router)
app.include_router(activation_contexts_router)


def cli(port: int = 8000) -> None:
    """Run the server.

    Args:
        port: Port to serve on (default 8000)
    """
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "spd.app.backend.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    fire.Fire(cli)
