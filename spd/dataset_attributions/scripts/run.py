"""Worker script for dataset attribution computation.

Called by SLURM jobs submitted via spd-attributions, or run directly for non-SLURM environments.

Usage:
    # Single GPU
    python -m spd.dataset_attributions.scripts.run <path> --n_batches 1000

    # Multi-GPU (run in parallel)
    python -m spd.dataset_attributions.scripts.run <path> --n_batches 1000 --rank 0 --world_size 4
    python -m spd.dataset_attributions.scripts.run <path> --n_batches 1000 --rank 1 --world_size 4
    ...
    python -m spd.dataset_attributions.scripts.run <path> --merge
"""

from spd.dataset_attributions.harvest import (
    DatasetAttributionConfig,
    harvest_attributions,
    merge_attributions,
)


def main(
    wandb_path: str,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 0.0,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
) -> None:
    """Compute dataset attributions, or merge results.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process. If None, processes entire training dataset.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for filtering components. Components with mean_ci <= threshold
            are excluded. Default 0.0 includes all components.
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

    if world_size is not None:
        print(f"Distributed harvest: {wandb_path} (rank {rank}/{world_size})")
    else:
        print(f"Single-GPU harvest: {wandb_path}")

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
