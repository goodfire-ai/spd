"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <decomposition_id> --config_json '...' --harvest_subrun_id h-20260211_120000
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.config import AutointerpEvalConfig
from spd.autointerp.repo import InterpRepo
from spd.autointerp.scoring.detection import run_detection_scoring
from spd.autointerp.scoring.fuzzing import run_fuzzing_scoring
from spd.harvest.repo import HarvestRepo

LabelScorerType = Literal["detection", "fuzzing"]


def main(
    decomposition_id: str,
    scorer_type: LabelScorerType,
    config_json: str,
    harvest_subrun_id: str,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    config = AutointerpEvalConfig.model_validate(config_json)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    interp_repo = InterpRepo.open(decomposition_id)
    assert interp_repo is not None, (
        f"No autointerp data for {decomposition_id}. Run autointerp first."
    )

    harvest = HarvestRepo(
        decomposition_id=decomposition_id,
        subrun_id=harvest_subrun_id,
        readonly=False,
    )

    components = harvest.get_all_components()

    match scorer_type:
        case "detection":
            asyncio.run(
                run_detection_scoring(
                    components=components,
                    interp_repo=interp_repo,
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    openrouter_api_key=openrouter_api_key,
                    tokenizer_name=tokenizer_name,
                    config=config.detection_config,
                    max_concurrent=config.max_concurrent,
                    max_requests_per_minute=config.max_requests_per_minute,
                    limit=config.limit,
                    cost_limit_usd=config.cost_limit_usd,
                )
            )
        case "fuzzing":
            asyncio.run(
                run_fuzzing_scoring(
                    components=components,
                    interp_repo=interp_repo,
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    openrouter_api_key=openrouter_api_key,
                    tokenizer_name=tokenizer_name,
                    config=config.fuzzing_config,
                    max_concurrent=config.max_concurrent,
                    max_requests_per_minute=config.max_requests_per_minute,
                    limit=config.limit,
                    cost_limit_usd=config.cost_limit_usd,
                )
            )


def get_command(
    decomposition_id: str,
    scorer_type: LabelScorerType,
    config: AutointerpEvalConfig,
    harvest_subrun_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.autointerp.scoring.scripts.run_label_scoring "
        f"--decomposition_id {decomposition_id} "
        f"--scorer_type {scorer_type} "
        f"--config_json '{config_json}' "
    )
    if harvest_subrun_id is not None:
        cmd += f" --harvest_subrun_id {harvest_subrun_id} "
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)
