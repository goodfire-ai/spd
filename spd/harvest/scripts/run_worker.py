"""Harvest worker: collects component statistics on a single GPU.

Usage:
    python -m spd.harvest.scripts.run_worker <wandb_path> --config_json '{"n_batches": 100}'
    python -m spd.harvest.scripts.run_worker <wandb_path> --config_json '...' --rank 0 --world_size 4 --subrun_id h-20260211_120000
"""

from datetime import datetime

import fire

from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import harvest_activation_contexts
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_json: str | dict[str, object] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    subrun_id: str | None = None,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)

    if subrun_id is None:
        subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = get_harvest_subrun_dir(run_id, subrun_id)

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    match config_json:
        case str(json_str):
            config = HarvestConfig.model_validate_json(json_str)
        case dict(d):
            config = HarvestConfig.model_validate(d)
        case None:
            config = HarvestConfig()

    if world_size is not None:
        logger.info(
            f"Distributed harvest: {wandb_path} (rank {rank}/{world_size}, subrun {subrun_id})"
        )
    else:
        logger.info(f"Single-GPU harvest: {wandb_path} (subrun {subrun_id})")

    harvest_activation_contexts(wandb_path, config, output_dir, rank, world_size)


if __name__ == "__main__":
    fire.Fire(main)
