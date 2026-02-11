"""CLI for autointerp pipeline.

Usage (direct execution):
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_path path/to/config.yaml
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '...'

Usage (SLURM submission):
    spd-autointerp <wandb_path>
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.repo import HarvestRepo
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_path: str | None = None,
    config_json: str | dict[str, object] | None = None,
    autointerp_run_id: str | None = None,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> None:
    """Interpret harvested components.

    Args:
        wandb_path: WandB run path (e.g. wandb:entity/project/runs/run_id).
        config_path: Path to CompactSkepticalConfig YAML/JSON file.
        config_json: Inline CompactSkepticalConfig as JSON string.
        autointerp_run_id: Pre-assigned run ID (timestamp). Generated if not provided.
        limit: Max number of components to interpret.
        cost_limit_usd: Cost budget in USD.
    """

    match (config_path, config_json):
        case (str(path), None):
            interp_config = CompactSkepticalConfig.from_file(path)
        case (None, str(json_str)):
            interp_config = CompactSkepticalConfig.model_validate_json(json_str)
        case (None, dict(d)):
            interp_config = CompactSkepticalConfig.model_validate(d)
        case _:
            raise ValueError("Exactly one of config_path or config_json must be provided")

    _, _, run_id = parse_wandb_run_path(wandb_path)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    harvest = HarvestRepo(run_id)

    # Create timestamped run directory
    if autointerp_run_id is None:
        autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_autointerp_dir(run_id) / autointerp_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(run_dir / "config.yaml")

    # DB lives at autointerp/<run_id>/interp.db (shared across autointerp runs)
    db_path = get_autointerp_dir(run_id) / "interp.db"

    print(f"Autointerp run: {run_dir}")

    run_interpret(
        wandb_path,
        openrouter_api_key,
        interp_config,
        harvest,
        db_path,
        limit,
        cost_limit_usd,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
