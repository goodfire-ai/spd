"""Worker script for harvest pipeline.

Usage (non-SLURM):
    # Single GPU
    python -m spd.harvest.scripts.run <wandb_path>

    # With config file
    python -m spd.harvest.scripts.run <wandb_path> --config_path path/to/config.yaml

    # Multi-GPU (run in parallel via shell, tmux, etc.)
    python -m spd.harvest.scripts.run <path> --config_json '...' --rank 0 --world_size 4 --subrun_id h-20260211_120000 &
    python -m spd.harvest.scripts.run <path> --config_json '...' --rank 1 --world_size 4 --subrun_id h-20260211_120000 &
    ...
    python -m spd.harvest.scripts.run <path> --merge --subrun_id h-20260211_120000

Usage (SLURM submission):
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8
"""

from datetime import datetime

from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import (
    harvest_activation_contexts,
    merge_activation_contexts,
)
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_path: str | None = None,
    config_json: str | dict[str, object] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
    subrun_id: str | None = None,
) -> None:
    """Harvest correlations and activation contexts, or merge results.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config_path: Path to HarvestConfig YAML/JSON file.
        config_json: Inline HarvestConfig as JSON string.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
        merge: If True, merge partial results from workers.
        subrun_id: Sub-run identifier (e.g. "h-20260211_120000"). Generated if not provided.
    """

    _, _, run_id = parse_wandb_run_path(wandb_path)

    if subrun_id is None:
        subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = get_harvest_subrun_dir(run_id, subrun_id)

    if merge:
        assert rank is None and world_size is None, "Cannot specify rank/world_size with --merge"
        print(f"Merging harvest results for {wandb_path} (subrun {subrun_id})")
        merge_activation_contexts(output_dir)
        return

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    match (config_path, config_json):
        case (str(path), None):
            config = HarvestConfig.from_file(path)
        case (None, str(json_str)):
            config = HarvestConfig.model_validate_json(json_str)
        case (None, dict(d)):
            config = HarvestConfig.model_validate(d)
        case (None, None):
            config = HarvestConfig()
        case _:
            raise ValueError("config_path and config_json are mutually exclusive")

    if world_size is not None:
        print(f"Distributed harvest: {wandb_path} (rank {rank}/{world_size}, subrun {subrun_id})")
    else:
        print(f"Single-GPU harvest: {wandb_path} (subrun {subrun_id})")

    harvest_activation_contexts(wandb_path, config, output_dir, rank, world_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
