"""Harvest merge: combines worker states into final harvest results.

Usage:
    python -m spd.harvest.scripts.run_merge <wandb_path> --subrun_id h-20260211_120000
"""

import fire

from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import merge_harvest
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    subrun_id: str,
    config_json: str | dict[str, object] | None = None,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)
    output_dir = get_harvest_subrun_dir(run_id, subrun_id)

    match config_json:
        case str(json_str):
            config = HarvestConfig.model_validate_json(json_str)
        case dict(d):
            config = HarvestConfig.model_validate(d)
        case None:
            config = HarvestConfig()

    logger.info(f"Merging harvest results for {wandb_path} (subrun {subrun_id})")
    merge_harvest(output_dir, config)


if __name__ == "__main__":
    fire.Fire(main)
