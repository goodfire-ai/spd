"""CLI for autointerp pipeline.

Usage (direct execution):
    python -m spd.autointerp.scripts.run_interpret <wandb_path>

Usage (SLURM submission):
    spd-autointerp <wandb_path>
"""

import os

from dotenv import load_dotenv

from spd.autointerp.interpret import OpenRouterModelName, run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.schemas import get_activation_contexts_dir, get_correlations_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    model: OpenRouterModelName,
    limit: int | None = None,
) -> None:
    """Interpret harvested components.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        model: OpenRouter model to use for interpretation.
        limit: Maximum number of components to interpret (highest mean CI first).
    """
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
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
