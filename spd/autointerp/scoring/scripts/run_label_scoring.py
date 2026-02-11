"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer detection
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer fuzzing
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer detection --autointerp_run_id 20260206_153040
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from spd.autointerp.db import InterpDB
from spd.autointerp.interpret import get_architecture_info
from spd.autointerp.repo import InterpRepo
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.repo import HarvestRepo
from spd.utils.wandb_utils import parse_wandb_run_path

LabelScorerType = Literal["detection", "fuzzing"]


def main(
    wandb_path: str,
    scorer: LabelScorerType,
    model: str = "google/gemini-3-flash-preview",
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    arch = get_architecture_info(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)

    interp = InterpRepo(run_id)
    interpretations = interp.get_all_interpretations()
    assert interpretations, f"No interpretation results for {run_id}. Run autointerp first."
    labels = {key: result.label for key, result in interpretations.items()}

    harvest = HarvestRepo(run_id)
    components = harvest.get_all_components()
    ci_threshold = harvest.get_ci_threshold()

    db_path = get_autointerp_dir(run_id) / "interp.db"
    db = InterpDB(db_path)

    match scorer:
        case "detection":
            from spd.autointerp.scoring.detection import run_detection_scoring

            asyncio.run(
                run_detection_scoring(
                    components=components,
                    labels=labels,
                    model=model,
                    openrouter_api_key=openrouter_api_key,
                    tokenizer_name=arch.tokenizer_name,
                    db=db,
                    limit=limit,
                    cost_limit_usd=cost_limit_usd,
                )
            )
        case "fuzzing":
            from spd.autointerp.scoring.fuzzing import run_fuzzing_scoring

            asyncio.run(
                run_fuzzing_scoring(
                    components=components,
                    labels=labels,
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
