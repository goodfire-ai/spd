"""Worker script for dataset attribution computation.

Called by SLURM jobs submitted via spd-attributions, or run directly for non-SLURM environments.

Usage:
    # Single GPU
    python -m spd.dataset_attributions.scripts.run <path>

    # With config file
    python -m spd.dataset_attributions.scripts.run <path> --config_path path/to/config.yaml

    # Multi-GPU (run in parallel)
    python -m spd.dataset_attributions.scripts.run <path> --config_json '...' --rank 0 --world_size 4
    ...
    python -m spd.dataset_attributions.scripts.run <path> --merge
"""

from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvest import (
    harvest_attributions,
    merge_attributions,
)


def main(
    wandb_path: str,
    config_path: str | None = None,
    config_json: str | dict[str, object] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
) -> None:
    """Compute dataset attributions, or merge results.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config_path: Path to DatasetAttributionConfig YAML/JSON file.
        config_json: Inline DatasetAttributionConfig as JSON string.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
        merge: If True, merge partial results from workers.
    """
    if merge:
        assert rank is None and world_size is None, "Cannot specify rank/world_size with --merge"
        print(f"Merging attribution results for {wandb_path}")
        merge_attributions(wandb_path)
        return

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    match (config_path, config_json):
        case (str(path), None):
            config = DatasetAttributionConfig.from_file(path)
        case (None, str(json_str)):
            config = DatasetAttributionConfig.model_validate_json(json_str)
        case (None, dict(d)):
            config = DatasetAttributionConfig.model_validate(d)
        case (None, None):
            config = DatasetAttributionConfig()
        case _:
            raise ValueError("config_path and config_json are mutually exclusive")

    if world_size is not None:
        print(f"Distributed harvest: {wandb_path} (rank {rank}/{world_size})")
    else:
        print(f"Single-GPU harvest: {wandb_path}")

    harvest_attributions(wandb_path, config, rank=rank, world_size=world_size)


def cli() -> None:
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
