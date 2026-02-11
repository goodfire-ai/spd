"""CLI for intruder detection eval.

Usage:
    python -m spd.autointerp.eval.scripts.run_intruder <wandb_path> --limit 100
    python -m spd.autointerp.eval.scripts.run_intruder <wandb_path> --harvest_subrun_id h-20260211_120000
"""

import asyncio
import os

from dotenv import load_dotenv

from spd.autointerp.eval.intruder import run_intruder_scoring
from spd.autointerp.interpret import get_architecture_info
from spd.harvest.db import HarvestDB
from spd.harvest.repo import HarvestRepo
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    model: str = "google/gemini-3-flash-preview",
    limit: int | None = None,
    cost_limit_usd: float | None = None,
    harvest_subrun_id: str | None = None,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    arch = get_architecture_info(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)

    harvest = HarvestRepo(run_id, subrun_id=harvest_subrun_id)
    components = harvest.get_all_components()
    ci_threshold = harvest.get_ci_threshold()

    db_path = harvest._resolve_db_path()
    assert db_path is not None, f"No harvest.db for run {run_id}"
    db = HarvestDB(db_path)

    asyncio.run(
        run_intruder_scoring(
            components=components,
            model=model,
            openrouter_api_key=openrouter_api_key,
            tokenizer_name=arch.tokenizer_name,
            db=db,
            ci_threshold=ci_threshold,
            limit=limit,
            cost_limit_usd=cost_limit_usd,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
