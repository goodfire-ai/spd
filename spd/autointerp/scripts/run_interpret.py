"""CLI for autointerp pipeline.

Usage:
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '...'
    spd-autointerp <wandb_path>  # SLURM submission
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.repo import HarvestRepo
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_json: str | dict[str, object],
    autointerp_run_id: str | None = None,
    harvest_subrun_id: str | None = None,
) -> None:
    match config_json:
        case str(json_str):
            interp_config = CompactSkepticalConfig.model_validate_json(json_str)
        case dict(d):
            interp_config = CompactSkepticalConfig.model_validate(d)

    _, _, run_id = parse_wandb_run_path(wandb_path)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    harvest = HarvestRepo.open(run_id, subrun_id=harvest_subrun_id)
    assert harvest is not None, f"No harvest data for {run_id}"

    # Create timestamped run directory
    if autointerp_run_id is None:
        autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_autointerp_dir(run_id) / autointerp_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(run_dir / "config.yaml")

    db_path = run_dir / "interp.db"

    logger.info(f"Autointerp run: {run_dir}")

    run_interpret(
        wandb_path,
        openrouter_api_key,
        interp_config,
        harvest,
        db_path,
        interp_config.limit,
        interp_config.cost_limit_usd,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
