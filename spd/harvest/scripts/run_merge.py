"""Harvest merge: combines worker states into final harvest results.

Usage:
    python -m spd.harvest.scripts.run_merge <wandb_path> --subrun_id h-20260211_120000
"""

import fire

from spd.decomposition.dispatch import decomposition_from_config
from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import merge_harvest
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger


def main(
    wandb_path: str,
    subrun_id: str,
    config_json: str,
) -> None:
    config = HarvestConfig.model_validate_json(config_json)
    decomposition = decomposition_from_config(config.target_decomposition)
    output_dir = get_harvest_subrun_dir(decomposition.id, subrun_id)
    logger.info(f"Merging harvest results for {wandb_path} (subrun {subrun_id})")
    merge_harvest(output_dir, config)


def get_command(subrun_id: str, config: HarvestConfig) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.harvest.scripts.run_merge "
        f"--subrun_id {subrun_id} "
        f"--config_json '{config_json}'"
    )
    return cmd


if __name__ == "__main__":
    fire.Fire(main)
