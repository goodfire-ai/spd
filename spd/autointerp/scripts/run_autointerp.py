"""CLI for autointerp pipeline.

Usage:
    # Harvest (single GPU)
    python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> --n_batches 1000

    # Harvest (parallel across 8 GPUs)
    python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> --n_batches 8000 -d 8

    # Interpret (CPU)
    python -m spd.autointerp.scripts.run_autointerp interpret <wandb_path>
"""

import os

from spd.autointerp.interpret import OpenRouterModelName


def harvest_cmd(
    wandb_path: str,
    n_batches: int,
    d: int | None = None,
    batch_size: int = 256,
    context_length: int = 512,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 100,
    activation_context_tokens_per_side: int = 10,
) -> None:
    """Harvest correlations and activation contexts.

    Args:
        d: Number of GPUs for distributed harvesting. If None, uses single GPU.
    """
    from spd.autointerp.harvest import HarvestConfig, harvest, harvest_parallel, save_harvest
    from spd.utils.wandb_utils import parse_wandb_run_path

    entity, project, run_id = parse_wandb_run_path(wandb_path)
    clean_path = f"{entity}/{project}/{run_id}"

    config = HarvestConfig(
        wandb_path=clean_path,
        n_batches=n_batches,
        batch_size=batch_size,
        context_length=context_length,
        ci_threshold=ci_threshold,
        activation_examples_per_component=activation_examples_per_component,
        activation_context_tokens_per_side=activation_context_tokens_per_side,
    )

    if d is not None:
        print(f"Distributed harvest: {clean_path} with {d} GPUs")
        result = harvest_parallel(config, d)
    else:
        print(f"Single-GPU harvest: {clean_path}")
        result = harvest(config)

    out_dir = save_harvest(result, run_id)
    print(f"Saved {len(result.components)} components to {out_dir}")


def interpret_cmd(
    wandb_path: str,
    model: OpenRouterModelName = OpenRouterModelName.GEMINI_2_5_FLASH,
    max_concurrent: int = 20,
) -> None:
    from spd.autointerp.interpret import run_interpret

    """Interpret harvested components."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY not set"
    run_interpret(wandb_path, api_key, model, max_concurrent)


if __name__ == "__main__":
    import fire

    fire.Fire({"harvest": harvest_cmd, "interpret": interpret_cmd})
