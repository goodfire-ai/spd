"""One-off script to harvest component correlations for a run.

Usage:
    python -m spd.app.scripts.harvest_correlations <wandb_path> [options]

Example:
    python -m spd.app.scripts.harvest_correlations anthropic/spd/abc123 --n_batches 500
"""

import fire
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.app.backend.lib.component_correlations import (
    get_correlations_path,
    harvest_correlations,
)
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    n_batches: int = 500,
    batch_size: int = 32,
    context_length: int = 128,
    ci_threshold: float = 1e-6,
) -> None:
    """Harvest component correlations for a run.

    Args:
        wandb_path: W&B run path (entity/project/run_id)
        n_batches: Number of batches to process
        batch_size: Batch size
        context_length: Context length
        ci_threshold: CI threshold for binarization
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Parse and normalize wandb path
    entity, project, run_id = parse_wandb_run_path(wandb_path)
    clean_wandb_path = f"{entity}/{project}/{run_id}"
    logger.info(f"Loading run: {clean_wandb_path}")

    # Load model
    run_info = SPDRunInfo.from_path(clean_wandb_path)
    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()

    # Load config
    spd_config = run_info.config
    assert spd_config.tokenizer_name is not None

    # Load tokenizer (needed for dataset)
    logger.info(f"Loading tokenizer: {spd_config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # Create data loader
    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)

    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=context_length,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
    )

    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed,
    )

    # Harvest correlations
    logger.info(
        f"Harvesting correlations: {n_batches} batches × {batch_size} batch_size "
        f"× {context_length} context_length = {n_batches * batch_size * context_length:,} tokens"
    )

    correlations = harvest_correlations(
        config=spd_config,
        cm=model,
        train_loader=train_loader,
        ci_threshold=ci_threshold,
        n_batches=n_batches,
    )

    # Save
    output_path = get_correlations_path(run_id)
    correlations.save(output_path)

    logger.info(f"Done! Correlations saved to {output_path}")
    logger.info(f"  - Components: {len(correlations.component_keys)}")
    logger.info(f"  - Tokens processed: {correlations.n_tokens:,}")

    # Quick sanity check: show top correlations for first active component
    active_components = [
        k for i, k in enumerate(correlations.component_keys) if correlations.count_i[i] > 0
    ]
    if active_components:
        test_key = active_components[0]
        top_corr = correlations.get_correlated(test_key, metric="f1", top_k=5)
        logger.info(f"  - Sample correlations for {test_key}:")
        for c in top_corr:
            logger.info(f"      {c.component_key}: F1={c.score:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
