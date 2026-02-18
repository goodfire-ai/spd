"""CLI for autointerp pipeline.

Usage:
    python -m spd.autointerp.scripts.run_interpret <decomposition_id> --config_json '...'
    spd-autointerp <decomposition_id>  # SLURM submission
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_dir
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    config_json: str | dict[str, object],
    autointerp_run_id: str | None = None,
    harvest_subrun_id: str | None = None,
) -> None:
    match config_json:
        case str(json_str):
            interp_config = CompactSkepticalConfig.model_validate_json(json_str)
        case dict(d):
            interp_config = CompactSkepticalConfig.model_validate(d)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(decomposition_id)
        assert harvest is not None, f"No harvest data for {decomposition_id}"

    # Create timestamped run directory
    if autointerp_run_id is None:
        autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_autointerp_dir(decomposition_id) / autointerp_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(run_dir / "config.yaml")

    db_path = run_dir / "interp.db"

    logger.info(f"Autointerp run: {run_dir}")

    run_interpret(
        decomposition_id,
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
