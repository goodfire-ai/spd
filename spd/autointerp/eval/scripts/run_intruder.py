"""CLI for intruder detection eval.

Usage:
    python -m spd.autointerp.eval.scripts.run_intruder <wandb_path> --limit 100
"""

import asyncio
import os

from dotenv import load_dotenv

from spd.autointerp.db import InterpDB
from spd.autointerp.eval.intruder import run_intruder_scoring
from spd.autointerp.interpret import get_architecture_info
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.loaders import load_all_components, load_harvest_ci_threshold
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    model: str = "google/gemini-3-flash-preview",
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    arch = get_architecture_info(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)

    components = load_all_components(run_id)
    ci_threshold = load_harvest_ci_threshold(run_id)

    db_path = get_autointerp_dir(run_id) / "interp.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = InterpDB(db_path)

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
