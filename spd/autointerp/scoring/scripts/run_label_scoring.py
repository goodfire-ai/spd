"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer detection --eval_config_json '...'
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer fuzzing --eval_config_json '...'
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from spd.autointerp.config import AutointerpEvalConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.interpret import get_architecture_info
from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo
from spd.utils.wandb_utils import parse_wandb_run_path

LabelScorerType = Literal["detection", "fuzzing"]


def main(
    wandb_path: str,
    scorer: LabelScorerType,
    eval_config_json: str,
    harvest_subrun_id: str | None = None,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    eval_config = AutointerpEvalConfig.model_validate_json(eval_config_json)

    arch = get_architecture_info(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)

    interp = InterpRepo.open(run_id)
    assert interp is not None, f"No autointerp data for {run_id}. Run autointerp first."
    interpretations = interp.get_all_interpretations()
    labels = {key: result.label for key, result in interpretations.items()}

    harvest = HarvestRepo.open(run_id, subrun_id=harvest_subrun_id)
    assert harvest is not None, f"No harvest data for {run_id}"
    components = harvest.get_all_components()
    ci_threshold = harvest.get_ci_threshold()

    subrun_dir = InterpRepo._find_latest_subrun_dir(run_id)
    assert subrun_dir is not None, f"No autointerp subrun found for {run_id}"
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
                    tokenizer_name=arch.tokenizer_name,
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
                    tokenizer_name=arch.tokenizer_name,
                    db=db,
                    ci_threshold=ci_threshold,
                    eval_config=eval_config,
                    limit=eval_config.limit,
                    cost_limit_usd=eval_config.cost_limit_usd,
                )
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
