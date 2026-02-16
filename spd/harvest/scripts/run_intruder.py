"""CLI for intruder detection eval.

Usage:
    python -m spd.harvest.scripts.run_intruder <wandb_path> --eval_config_json '...' --harvest_subrun_id h-20260211_120000
"""

import asyncio
import os

from dotenv import load_dotenv

from spd.autointerp.interpret import get_architecture_info
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.db import HarvestDB
from spd.harvest.intruder import run_intruder_scoring
from spd.harvest.repo import HarvestRepo
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    eval_config_json: str | dict[str, object],
    harvest_subrun_id: str,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    match eval_config_json:
        case str(json_str):
            eval_config = IntruderEvalConfig.model_validate_json(json_str)
        case dict(d):
            eval_config = IntruderEvalConfig.model_validate(d)

    arch = get_architecture_info(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)

    harvest = HarvestRepo.open(run_id, subrun_id=harvest_subrun_id)
    assert harvest is not None, f"No harvest data for {run_id}"
    components = harvest.get_all_components()
    ci_threshold = harvest.get_activation_threshold()

    db = HarvestDB(harvest.db_path)

    asyncio.run(
        run_intruder_scoring(
            components=components,
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
