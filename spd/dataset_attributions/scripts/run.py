"""CLI for dataset attribution computation.

Usage:
    spd-attributions <wandb_path> --n_batches 1000
    spd-attributions <wandb_path> --n_batches 1000 --batch_size 32
"""

from spd.dataset_attributions.harvest import DatasetAttributionConfig, harvest_attributions


def main(
    wandb_path: str,
    n_batches: int,
    batch_size: int = 64,
    ci_threshold: float = 1e-6,
) -> None:
    """Compute dataset attributions.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
    """
    print(f"Computing dataset attributions: {n_batches} batches, batch_size={batch_size}")

    config = DatasetAttributionConfig(
        wandb_path=wandb_path,
        n_batches=n_batches,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
    )
    harvest_attributions(config)


def cli() -> None:
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
