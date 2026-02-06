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
from spd.harvest.harvest import HarvestResult
from spd.harvest.schemas import get_activation_contexts_dir
from spd.settings import SPD_OUT_DIR


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

    activation_contexts_dir = get_activation_contexts_dir(run_id)
    assert activation_contexts_dir.exists(), f"No harvest data at {activation_contexts_dir}"

    components = HarvestResult.load_components(activation_contexts_dir)

    scoring_dir = SPD_OUT_DIR / "autointerp" / run_id / "eval" / "intruder"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = scoring_dir / f"results_{timestamp}.jsonl"

    asyncio.run(
        run_intruder_scoring(
            components=components,
            model=model,
            openrouter_api_key=openrouter_api_key,
            tokenizer_name=arch.tokenizer_name,
            output_path=output_path,
            limit=limit,
            cost_limit_usd=cost_limit_usd,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
