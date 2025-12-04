"""FastAPI routers for the SPD backend API."""

from spd.app.backend.routers.activation_contexts import router as activation_contexts_router
from spd.app.backend.routers.compute import router as compute_router
from spd.app.backend.routers.graphs import router as graphs_router
from spd.app.backend.routers.runs import router as runs_router

__all__ = [
    "activation_contexts_router",
    "compute_router",
    "graphs_router",
    "runs_router",
]
