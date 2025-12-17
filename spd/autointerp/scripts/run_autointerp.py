"""CLI for autointerp pipeline.

Usage (direct execution):
    # Harvest (single GPU)
    python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> --n_batches 1000

    # Harvest (parallel across 8 GPUs)
    python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> --n_batches 8000 --n_gpus 8

    # Interpret (CPU)
    python -m spd.autointerp.scripts.run_autointerp interpret <wandb_path>

Usage (SLURM submission):
    # Submit harvest job to SLURM
    spd-harvest <wandb_path> --n_batches 1000

    # Submit interpret job to SLURM
    spd-interpret <wandb_path>
"""

import os

from spd.autointerp.interpret import OpenRouterModelName
from spd.autointerp.schemas import (
    get_activation_contexts_dir,
    get_autointerp_dir,
    get_correlations_dir,
)
from spd.utils.wandb_utils import parse_wandb_run_path


def harvest_cmd(
    wandb_path: str,
    n_batches: int,
    n_gpus: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
) -> None:
    """Harvest correlations and activation contexts.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process.
        n_gpus: Number of GPUs for distributed harvesting. If None, uses single GPU.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI in include
    """
    from spd.autointerp.harvest import HarvestConfig, harvest, harvest_parallel

    _, _, run_id = parse_wandb_run_path(wandb_path)

    config = HarvestConfig(
        wandb_path=wandb_path,
        n_batches=n_batches,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
        activation_examples_per_component=activation_examples_per_component,
        activation_context_tokens_per_side=activation_context_tokens_per_side,
        pmi_token_top_k=pmi_token_top_k,
    )

    # Output directories for harvest results
    activation_contexts_dir = get_activation_contexts_dir(run_id)
    correlations_dir = get_correlations_dir(run_id)

    if n_gpus is not None:
        print(f"Distributed harvest: {wandb_path} with {n_gpus} GPUs")
        harvest_parallel(config, n_gpus, activation_contexts_dir, correlations_dir)
    else:
        print(f"Single-GPU harvest: {wandb_path}")
        harvest(config, activation_contexts_dir, correlations_dir)


def interpret_cmd(
    wandb_path: str,
    model: OpenRouterModelName = OpenRouterModelName.GEMINI_2_5_FLASH,
    max_concurrent: int = 20,
    budget: float | None = None,
) -> None:
    """Interpret harvested components.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        model: OpenRouter model to use for interpretation.
        max_concurrent: Maximum concurrent API requests.
        budget: Stop after spending this much (USD). None = unlimited.
    """

    from spd.autointerp.interpret import run_interpret

    _, _, run_id = parse_wandb_run_path(wandb_path)

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    activation_contexts_dir = get_activation_contexts_dir(run_id)
    assert activation_contexts_dir.exists(), (
        f"Activation contexts not found at {activation_contexts_dir}. Run harvest first."
    )

    autointerp_dir = get_autointerp_dir(run_id)
    autointerp_dir.mkdir(parents=True, exist_ok=True)

    run_interpret(
        wandb_path,
        openrouter_api_key,
        model,
        max_concurrent,
        activation_contexts_dir,
        autointerp_dir,
        budget,
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"harvest": harvest_cmd, "interpret": interpret_cmd})
