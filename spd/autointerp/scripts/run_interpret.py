"""CLI for autointerp pipeline.

Usage (direct execution):
    python -m spd.autointerp.scripts.run_interpret <wandb_path>
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config path/to/config.yaml

Usage (SLURM submission):
    spd-autointerp <wandb_path>
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.config import CompactSkepticalConfig, ReasoningEffort
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.schemas import (
    get_activation_contexts_dir,
    get_correlations_dir,
    load_harvest_ci_threshold,
)
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config: str | None = None,
    model: str = "google/gemini-3-flash-preview",
    reasoning_effort: str | None = None,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> None:
    """Interpret harvested components.

    Args:
        wandb_path: WandB run path (e.g. wandb:entity/project/runs/run_id).
        config: Path to AutointerpConfig YAML file. If provided, model/reasoning_effort are ignored.
        model: OpenRouter model name (used when config is not provided).
        reasoning_effort: Reasoning effort level (used when config is not provided).
        limit: Max number of components to interpret.
        cost_limit_usd: Cost budget in USD.
    """
    _, _, run_id = parse_wandb_run_path(wandb_path)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    # Build or load config
    if config is not None:
        interp_config = CompactSkepticalConfig.from_file(config)
    else:
        effort = ReasoningEffort(reasoning_effort) if reasoning_effort else None
        interp_config = CompactSkepticalConfig(model=model, reasoning_effort=effort)

    activation_contexts_dir = get_activation_contexts_dir(run_id)
    assert activation_contexts_dir.exists(), (
        f"Activation contexts not found at {activation_contexts_dir}. Run harvest first."
    )

    correlations_dir = get_correlations_dir(run_id)

    # Create timestamped run directory
    autointerp_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_autointerp_dir(run_id) / autointerp_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(run_dir / "config.yaml")
    output_path = run_dir / "results.jsonl"

    print(f"Autointerp run: {run_dir}")

    ci_threshold = load_harvest_ci_threshold(run_id)

    run_interpret(
        wandb_path,
        openrouter_api_key,
        interp_config,
        activation_contexts_dir,
        correlations_dir,
        output_path,
        ci_threshold,
        limit,
        cost_limit_usd,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
