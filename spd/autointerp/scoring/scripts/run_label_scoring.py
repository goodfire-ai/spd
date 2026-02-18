"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <decomposition_id> --scorer detection --eval_config_json '...'
    python -m spd.autointerp.scoring.scripts.run_label_scoring <decomposition_id> --scorer fuzzing --eval_config_json '...'
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.config import AutointerpEvalConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo

LabelScorerType = Literal["detection", "fuzzing"]


def main(
    decomposition_id: str,
    scorer: LabelScorerType,
    eval_config_json: str | dict[str, object],
    harvest_subrun_id: str | None = None,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    match eval_config_json:
        case str(json_str):
            eval_config = AutointerpEvalConfig.model_validate_json(json_str)
        case dict(d):
            eval_config = AutointerpEvalConfig.model_validate(d)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    interp = InterpRepo.open(decomposition_id)
    assert interp is not None, f"No autointerp data for {decomposition_id}. Run autointerp first."
    interpretations = interp.get_all_interpretations()
    labels = {key: result.label for key, result in interpretations.items()}

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(decomposition_id)
        assert harvest is not None, f"No harvest data for {decomposition_id}"
    components = harvest.get_all_components()

    subrun_dir = InterpRepo._find_latest_subrun_dir(decomposition_id)
    assert subrun_dir is not None, f"No autointerp subrun found for {decomposition_id}"
    db = InterpDB(subrun_dir / "interp.db")

    match scorer:
        case "detection":
            from spd.autointerp.scoring.detection import run_detection_scoring

            asyncio.run(
                run_detection_scoring(
                    components=components,
                    labels=labels,
                    model=eval_config.model,
                    reasoning_effort=eval_config.reasoning_effort,
                    openrouter_api_key=openrouter_api_key,
                    tokenizer_name=tokenizer_name,
                    db=db,
                    eval_config=eval_config,
                    limit=eval_config.limit,
                    cost_limit_usd=eval_config.cost_limit_usd,
                )
            )
        case "fuzzing":
            from spd.autointerp.scoring.fuzzing import run_fuzzing_scoring

            asyncio.run(
                run_fuzzing_scoring(
                    components=components,
                    labels=labels,
                    model=eval_config.model,
                    openrouter_api_key=openrouter_api_key,
                    tokenizer_name=tokenizer_name,
                    db=db,
                    eval_config=eval_config,
                    limit=eval_config.limit,
                    cost_limit_usd=eval_config.cost_limit_usd,
                )
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
