"""Worker script for dataset attribution computation.

Called by SLURM jobs submitted via spd-attributions, or run directly for non-SLURM environments.

Usage:
    # Single GPU
    python -m spd.dataset_attributions.scripts.run <path> --config_json '...'

    # Multi-GPU (run in parallel)
    python -m spd.dataset_attributions.scripts.run <path> --config_json '...' --rank 0 --world_size 4 --subrun_id da-20260211_120000
    ...
    python -m spd.dataset_attributions.scripts.run <path> --merge --subrun_id da-20260211_120000
"""

from datetime import datetime

from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvest import (
    harvest_attributions,
    merge_attributions,
)
from spd.dataset_attributions.repo import get_attributions_subrun_dir
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_json: str,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
    subrun_id: str | None = None,
    harvest_subrun_id: str | None = None,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)

    if subrun_id is None:
        subrun_id = "da-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = get_attributions_subrun_dir(run_id, subrun_id)

    if merge:
        assert rank is None and world_size is None, "Cannot specify rank/world_size with --merge"
        logger.info(f"Merging attribution results for {wandb_path} (subrun {subrun_id})")
        merge_attributions(output_dir)
        return

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    config = DatasetAttributionConfig.from_json_or_dict(config_json)

    if world_size is not None:
        logger.info(
            f"Distributed harvest: {wandb_path} (rank {rank}/{world_size}, subrun {subrun_id})"
        )
    else:
        logger.info(f"Single-GPU harvest: {wandb_path} (subrun {subrun_id})")

    harvest_attributions(
        wandb_path=wandb_path,
        config=config,
        output_dir=output_dir,
        harvest_subrun_id=harvest_subrun_id,
        rank=rank,
        world_size=world_size,
    )


def get_worker_command(
    wandb_path: str,
    config_json: str,
    rank: int,
    world_size: int,
    subrun_id: str,
    harvest_subrun_id: str | None = None,
) -> str:
    cmd = (
        f"python -m spd.dataset_attributions.scripts.run "
        f'"{wandb_path}" '
        f"--config_json '{config_json}' "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id}"
    )
    if harvest_subrun_id is not None:
        cmd += f" --harvest_subrun_id {harvest_subrun_id}"
    return cmd


def get_merge_command(wandb_path: str, subrun_id: str) -> str:
    return (
        f"python -m spd.dataset_attributions.scripts.run "
        f'"{wandb_path}" '
        "--merge "
        f"--subrun_id {subrun_id}"
    )


def cli() -> None:
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
