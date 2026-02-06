"""CLI for autointerp pipeline.

Usage (direct execution):
    python -m spd.autointerp.scripts.run_interpret <wandb_path>

Usage (SLURM submission):
    spd-autointerp <wandb_path>
"""

import os

from dotenv import load_dotenv

from spd.autointerp.interpret import OpenRouterModelName, ReasoningEffort, run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.schemas import get_activation_contexts_dir, get_correlations_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    model: OpenRouterModelName = OpenRouterModelName.GEMINI_3_FLASH_PREVIEW,
    limit: int | None = None,
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW,
    cost_limit_usd: float | None = None,
) -> None:
    """Interpret harvested components."""
    _, _, run_id = parse_wandb_run_path(wandb_path)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    activation_contexts_dir = get_activation_contexts_dir(run_id)
    assert activation_contexts_dir.exists(), (
        f"Activation contexts not found at {activation_contexts_dir}. Run harvest first."
    )

    correlations_dir = get_correlations_dir(run_id)

    autointerp_dir = get_autointerp_dir(run_id)
    autointerp_dir.mkdir(parents=True, exist_ok=True)

    run_interpret(
        wandb_path,
        openrouter_api_key,
        model,
        activation_contexts_dir,
        correlations_dir,
        autointerp_dir,
        limit,
        reasoning_effort,
        cost_limit_usd,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
