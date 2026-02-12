"""Data sources provenance endpoint.

Shows where harvest/autointerp data came from: subrun IDs, configs, counts.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors


class HarvestInfo(BaseModel):
    subrun_id: str
    config: dict[str, Any]
    n_components: int


class AutointerpInfo(BaseModel):
    subrun_id: str
    config: dict[str, Any]
    n_interpretations: int
    eval_scores: list[str]


class DataSourcesResponse(BaseModel):
    harvest: HarvestInfo | None
    autointerp: AutointerpInfo | None


router = APIRouter(prefix="/api/data_sources", tags=["data_sources"])


@router.get("")
@log_errors
def get_data_sources(loaded: DepLoadedRun) -> DataSourcesResponse:
    harvest_info: HarvestInfo | None = None
    if loaded.harvest is not None:
        harvest_info = HarvestInfo(
            subrun_id=loaded.harvest.subrun_id,
            config=loaded.harvest.get_config(),
            n_components=loaded.harvest.get_component_count(),
        )

    autointerp_info: AutointerpInfo | None = None
    if loaded.interp is not None:
        config = loaded.interp.get_config()
        if config is not None:
            # intruder scores live in harvest.db, detection/fuzzing in interp.db
            eval_scores = loaded.interp.get_available_score_types()
            if loaded.harvest is not None and loaded.harvest.get_intruder_scores() is not None:
                eval_scores = ["intruder", *eval_scores]

            autointerp_info = AutointerpInfo(
                subrun_id=loaded.interp.subrun_id,
                config=config,
                n_interpretations=loaded.interp.get_interpretation_count(),
                eval_scores=eval_scores,
            )

    return DataSourcesResponse(harvest=harvest_info, autointerp=autointerp_info)
