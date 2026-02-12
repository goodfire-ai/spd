"""Harvest merge: combines worker states into final harvest results.

Usage:
    python -m spd.harvest.scripts.run_merge <wandb_path> --subrun_id h-20260211_120000
"""

import fire

from spd.harvest.harvest import merge_activation_contexts
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(wandb_path: str, subrun_id: str) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)
    output_dir = get_harvest_subrun_dir(run_id, subrun_id)
    logger.info(f"Merging harvest results for {wandb_path} (subrun {subrun_id})")
    merge_activation_contexts(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
