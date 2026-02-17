"""Harvest worker: collects component statistics on a single GPU.

Usage:
    python -m spd.harvest.scripts.run_worker --config_json '{"n_batches": 100}'
    python -m spd.harvest.scripts.run_worker --config_json '...' --rank 0 --world_size 4 --subrun_id h-20260211_120000
"""

from datetime import datetime

import fire
import torch

from spd.decomposition.dispatch import decomposition_from_config
from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import harvest
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger
from spd.utils.distributed_utils import get_device


def main(
    config_json: str,
    rank_world_size: tuple[int, int] | None,
    subrun_id: str | None = None,
) -> None:
    if subrun_id is None:
        subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    config = HarvestConfig.model_validate_json(config_json)

    decomposition = decomposition_from_config(config.target_decomposition)

    if rank_world_size is not None:
        r, w = rank_world_size
        logger.info(f"Distributed harvest: rank {r}/{w}, subrun {subrun_id}")
    else:
        logger.info(f"Single-GPU harvest: subrun {subrun_id}")

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    output_dir = get_harvest_subrun_dir(decomposition.id, subrun_id)

    harvest(
        decomposition=decomposition,
        config=config,
        output_dir=output_dir,
        rank_world_size=rank_world_size,
        device=device,
    )


def get_command(config: HarvestConfig, rank: int, world_size: int, subrun_id: str) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.harvest.scripts.run_worker "
        f"--config_json '{config_json}' "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id}"
    )
    return cmd


if __name__ == "__main__":
    fire.Fire(main)
