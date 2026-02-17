"""CLI for autointerp pipeline.

Usage:
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '...'
    spd-autointerp <wandb_path>  # SLURM submission
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.config import AutointerpConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_subrun_dir
from spd.decomposition.dispatch import decomposition_from_id
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    harvest_subrun_id: str,
    config_json: str,
    autointerp_run_id: str | None = None,
) -> None:
    interp_config = AutointerpConfig.model_validate_json(config_json)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=False)

    # Create timestamped run directory
    if autointerp_run_id is None:
        autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    subrun_dir = get_autointerp_subrun_dir(decomposition_id, autointerp_run_id)
    subrun_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(subrun_dir / "config.yaml")

    db_path = subrun_dir / "interp.db"

    logger.info(f"Autointerp run: {subrun_dir}")

    decomposition = decomposition_from_id(decomposition_id)

    run_interpret(
        openrouter_api_key=openrouter_api_key,
        model=interp_config.model,
        reasoning_effort=interp_config.reasoning_effort,
        limit=interp_config.limit,
        cost_limit_usd=interp_config.cost_limit_usd,
        max_requests_per_minute=interp_config.max_requests_per_minute,
        arch=decomposition.architecture_info,
        template_strategy=interp_config.template_strategy,
        harvest=harvest,
        db_path=db_path,
    )


def get_command(
    decomposition_id: str,
    harvest_subrun_id: str,
    config: AutointerpConfig,
    autointerp_run_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    return (
        "python -m spd.autointerp.scripts.run_interpret "
        f"--decomposition_id {decomposition_id} "
        f"--harvest_subrun_id {harvest_subrun_id} "
        f"--config_json '{config_json}' "
        f"--autointerp_run_id {autointerp_run_id} "
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
