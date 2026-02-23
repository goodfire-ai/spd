"""Worker script for harvest pipeline.

Usage (non-SLURM):
    # Single GPU
    python -m spd.harvest.scripts.run <wandb_path> --n_batches 1000

    # Multi-GPU (run in parallel via shell, tmux, etc.)
    python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 0 --world_size 4 &
    python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 1 --world_size 4 &
    python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 2 --world_size 4 &
    python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 3 --world_size 4 &
    wait

    # Merge results after all workers complete
    python -m spd.harvest.scripts.run <path> --merge

Usage (SLURM submission):
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8
"""

from spd.harvest.harvest import (
    HarvestConfig,
    harvest_activation_contexts,
    merge_activation_contexts,
)
from spd.harvest.schemas import get_activation_contexts_dir, get_correlations_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
) -> None:
    """Harvest correlations and activation contexts, or merge results.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process. If None, processes entire training dataset.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI to include.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
        merge: If True, merge partial results from workers.
    """

    _, _, run_id = parse_wandb_run_path(wandb_path)

    if merge:
        assert rank is None and world_size is None, "Cannot specify rank/world_size with --merge"
        print(f"Merging harvest results for {wandb_path}")
        merge_activation_contexts(wandb_path)
        return

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    config = HarvestConfig(
        wandb_path=wandb_path,
        n_batches=n_batches,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
        activation_examples_per_component=activation_examples_per_component,
        activation_context_tokens_per_side=activation_context_tokens_per_side,
        pmi_token_top_k=pmi_token_top_k,
    )

    activation_contexts_dir = get_activation_contexts_dir(run_id)
    correlations_dir = get_correlations_dir(run_id)

    if world_size is not None:
        print(f"Distributed harvest: {wandb_path} (rank {rank}/{world_size})")
    else:
        print(f"Single-GPU harvest: {wandb_path}")

    harvest_activation_contexts(config, activation_contexts_dir, correlations_dir, rank, world_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
