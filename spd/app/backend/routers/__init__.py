"""FastAPI routers for the SPD backend API."""

from spd.app.backend.routers.activation_contexts import router as activation_contexts_router
from spd.app.backend.routers.agents import router as agents_router
from spd.app.backend.routers.clusters import router as clusters_router
from spd.app.backend.routers.component_data import router as component_data_router
from spd.app.backend.routers.correlations import router as correlations_router
from spd.app.backend.routers.data_sources import router as data_sources_router
from spd.app.backend.routers.dataset_attributions import router as dataset_attributions_router
from spd.app.backend.routers.dataset_search import router as dataset_search_router
from spd.app.backend.routers.graphs import router as graphs_router
from spd.app.backend.routers.intervention import router as intervention_router
from spd.app.backend.routers.pretrain_info import router as pretrain_info_router
from spd.app.backend.routers.prompts import router as prompts_router
from spd.app.backend.routers.runs import router as runs_router

__all__ = [
    "activation_contexts_router",
    "agents_router",
    "clusters_router",
    "component_data_router",
    "correlations_router",
    "data_sources_router",
    "dataset_attributions_router",
    "dataset_search_router",
    "graphs_router",
    "intervention_router",
    "pretrain_info_router",
    "prompts_router",
    "runs_router",
]
