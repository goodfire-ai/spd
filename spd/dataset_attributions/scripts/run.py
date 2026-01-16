"""CLI for dataset attribution computation.

Usage:
    # Single GPU
    spd-attributions <wandb_path> --n_batches 1000

    # Multi-GPU (launch workers separately, then merge)
    spd-attributions <path> --n_batches 1000 --rank 0 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 1 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 2 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 3 --world_size 4 &
    wait
    spd-attributions <path> --merge
"""

from spd.dataset_attributions.harvest import (
    DatasetAttributionConfig,
    harvest_attributions,
    merge_attributions,
)


def main(
    wandb_path: str,
    n_batches: int = 0,
    batch_size: int = 64,
    ci_threshold: float = 0.0,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
) -> None:
    """Compute dataset attributions.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for filtering components. Components with mean_ci <= threshold
            are excluded. Default 0.0 includes all components.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers for parallel execution.
        merge: If True, merge existing rank files instead of computing.
    """
    if merge:
        merge_attributions(wandb_path)
        return

    assert n_batches > 0, "--n_batches is required when not using --merge"

    if rank is not None:
        assert world_size is not None, "--world_size required with --rank"
        print(f"Worker {rank}/{world_size}: processing batches where idx % {world_size} == {rank}")
    else:
        print(f"Computing dataset attributions: {n_batches} batches, batch_size={batch_size}")

    config = DatasetAttributionConfig(
        wandb_path=wandb_path,
        n_batches=n_batches,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
    )

    harvest_attributions(config, rank=rank, world_size=world_size)


def cli() -> None:
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
