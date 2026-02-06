"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer detection
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer fuzzing
    python -m spd.autointerp.scoring.scripts.run_label_scoring <wandb_path> --scorer detection --autointerp_run_id 20260206_153040
"""

import asyncio
import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv

from spd.autointerp.interpret import get_architecture_info
from spd.autointerp.loaders import load_interpretations
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.harvest import HarvestResult
from spd.harvest.schemas import get_activation_contexts_dir, load_harvest_ci_threshold

LabelScorerType = Literal["detection", "fuzzing"]


def main(
    wandb_path: str,
    scorer: LabelScorerType,
    autointerp_run_id: str | None = None,
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

    interpretations = load_interpretations(run_id, autointerp_run_id)
    assert interpretations, f"No interpretation results for {run_id}. Run autointerp first."
    labels = {key: result.label for key, result in interpretations.items()}

    components = HarvestResult.load_components(activation_contexts_dir)
    ci_threshold = load_harvest_ci_threshold(run_id)

    # Scoring output goes under the autointerp run dir if specified, else under SPD run dir
    if autointerp_run_id is not None:
        scoring_dir = get_autointerp_dir(run_id) / autointerp_run_id / "scoring" / scorer
    else:
        scoring_dir = get_autointerp_dir(run_id) / "scoring" / scorer
    scoring_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = scoring_dir / f"results_{timestamp}.jsonl"

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
                    output_path=output_path,
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
                    output_path=output_path,
                    ci_threshold=ci_threshold,
                    limit=limit,
                    cost_limit_usd=cost_limit_usd,
                )
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
