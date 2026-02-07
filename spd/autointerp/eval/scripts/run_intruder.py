"""CLI for intruder detection eval.

Usage:
    python -m spd.autointerp.eval.scripts.run_intruder <wandb_path> --limit 100
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.eval.intruder import run_intruder_scoring
from spd.autointerp.interpret import get_architecture_info
from spd.harvest.loaders import load_all_components, load_harvest_ci_threshold
from spd.harvest.schemas import get_harvest_dir


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
    run_id = wandb_path.split("/")[-1]

    components = load_all_components(run_id)

    scoring_dir = get_harvest_dir(run_id) / "eval" / "intruder"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = scoring_dir / f"results_{timestamp}.jsonl"

    ci_threshold = load_harvest_ci_threshold(run_id)

    asyncio.run(
        run_intruder_scoring(
            components=components,
            model=model,
            openrouter_api_key=openrouter_api_key,
            tokenizer_name=arch.tokenizer_name,
            output_path=output_path,
            ci_threshold=ci_threshold,
            limit=limit,
            cost_limit_usd=cost_limit_usd,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
